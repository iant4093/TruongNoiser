#!/usr/bin/env python3
"""Independent official FP32 CGNet versus Core ML raw-tensor fidelity and timings.

Run one size, compute request and ensemble mode per bounded process. This is
conversion validation, not a SIDD score or physical-phone/ANE quality claim.
Off-domain reference excursions require refusal and never count as fidelity
passes. Every orientation is checked before averaging can conceal an excursion.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

from cgnet_reference import (
    ARCHITECTURE, CHECKPOINT_SHA256, COMMIT, checkpoint_manifest,
    load_official_model, sha256, verify_official_sources,
)

CASE_NAMES = (
    'black', 'white', 'near-black', 'gradient', 'high-frequency-fine-grain',
    'bounded-photo-noise', 'fine-color-stripes', 'diagonal-detail',
    'high-frequency-color', 'seeded-noise', 'official-noisy-demo',
)
COMPUTE_POLICIES = ('cpuOnly', 'cpuAndGPU', 'cpuAndNeuralEngine', 'all')


def compiled_identity(path):
    """Bind evidence to every actual compiled byte, using bounded streaming reads."""
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError('Compiled candidates must be real directories, not symbolic links')
    files = []
    digest = hashlib.sha256()
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise RuntimeError('Compiled candidates must not contain symbolic links')
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        value = sha256(item)
        files.append({'path': relative, 'bytes': item.stat().st_size, 'sha256': value})
        digest.update(relative.encode() + b'\0' + value.encode() + b'\n')
    if not files:
        raise RuntimeError('Empty compiled candidate')
    return digest.hexdigest(), files


def validate_compiled_tensor_contract(path, size):
    metadata = json.loads((path / 'metadata.json').read_text())
    if not isinstance(metadata, list) or len(metadata) != 1:
        raise RuntimeError('Expected one compiled Core ML model description')
    for kind, name in [('inputSchema', 'input_tile'), ('outputSchema', 'output_tile')]:
        schema = metadata[0].get(kind, [])
        if (len(schema) != 1 or schema[0].get('name') != name or
                schema[0].get('dataType') != 'Float16' or
                json.loads(schema[0].get('shape', 'null')) != [1, 3, size, size]):
            raise RuntimeError(f'Invalid declared trained tensor contract: {kind}')


def validate_prediction_bridge_dtype(actual, np):
    # Core ML's Python bridge promotes MLMultiArray Float16 outputs to NumPy Float32.
    # Check the compiled schema separately, and require every finite promoted sample
    # to be exactly representable as Float16; never silently round arbitrary Float32.
    if actual.dtype == np.float16:
        return
    if actual.dtype != np.float32:
        raise RuntimeError(f'Unexpected prediction bridge type: {actual.dtype}')
    finite = actual[np.isfinite(actual)]
    with np.errstate(over='ignore', invalid='ignore'):
        if not np.array_equal(finite, finite.astype(np.float16).astype(np.float32)):
            raise RuntimeError('Float32 bridge values do not preserve the declared Float16 lattice')


def make_cases(size, np, Image):
    root = Path(__file__).resolve().parents[1]
    demo_path = root / 'ModelConversionTools/official/NAFNet/demo/noisy.png'
    if sha256(demo_path) != '403034182fa320130dae0d75b92e85e0850771378e674d65455c403a4958e29c':
        raise RuntimeError('Official NAFNet noisy-photo fixture changed')
    demo = np.asarray(Image.open(demo_path).convert('RGB'), dtype=np.float32) / 255
    # Repeat the original noisy pixels to extend the retained256px photo for512;
    # never resize/smooth the reference fixture. This is not a512px SIDD score.
    demo = np.tile(demo, (math.ceil(size / demo.shape[0]), math.ceil(size / demo.shape[1]), 1))[:size, :size]
    demo = demo.transpose(2, 0, 1)[None].astype(np.float16)
    rng = np.random.default_rng(261029)
    yy, xx = np.indices((size, size))
    checker = ((xx + yy) % 2).astype(np.float16)
    gradient = np.broadcast_to(np.linspace(0, 1, size, dtype=np.float16), (size, size))
    stripes = (xx % 2).astype(np.float16)
    diagonal = ((xx + 2 * yy) % 7 < 3).astype(np.float16)
    return {
        'black': np.zeros((1, 3, size, size), np.float16),
        'white': np.ones((1, 3, size, size), np.float16),
        'near-black': np.full((1, 3, size, size), 1 / 1024, np.float16),
        'gradient': np.stack([gradient, gradient.T, gradient * .5])[None],
        'high-frequency-fine-grain': np.stack([.48 + .04 * checker] * 3)[None],
        'bounded-photo-noise': np.clip(.5 + rng.normal(0, .025, (1, 3, size, size)), 0, 1).astype(np.float16),
        'fine-color-stripes': np.stack([.45 + .1 * stripes, .55 - .1 * stripes, .5 + .025 * stripes])[None],
        'diagonal-detail': np.stack([.25 + .5 * diagonal, .35 + .3 * diagonal, .4 + .2 * diagonal])[None],
        'high-frequency-color': np.stack([checker, 1 - checker, checker * .5])[None],
        'seeded-noise': rng.uniform(0, 1, (1, 3, size, size)).astype(np.float16),
        'official-noisy-demo': demo,
    }


def oriented(value, quarter_turns, mirrored, np):
    transformed = np.flip(value, axis=3) if mirrored else value
    return np.ascontiguousarray(np.rot90(transformed, quarter_turns, axes=(2, 3)))


def undo_orientation(value, quarter_turns, mirrored, np):
    result = np.rot90(value, -quarter_turns, axes=(2, 3))
    return np.ascontiguousarray(np.flip(result, axis=3) if mirrored else result)


def raw_summary(value, np):
    finite = bool(np.isfinite(value).all())
    # Python's strict JSON cannot represent NaN/Inf. Keep their presence explicit
    # and report extrema only for finite elements, without substituting pixels.
    finite_values = value[np.isfinite(value)]
    return {
        'finite': finite,
        'minFinite': float(finite_values.min()) if finite_values.size else None,
        'maxFinite': float(finite_values.max()) if finite_values.size else None,
        'requiresProductionRefusal': not (finite and bool(np.all((value >= -.25) & (value <= 1.25)))),
    }


def raw_refusal_failure(reference, prediction):
    """Keep a raw failure visible even if another orientation requires reference refusal."""
    if reference['requiresProductionRefusal'] and not prediction['requiresProductionRefusal']:
        return 'masked-unsafe-reference'
    if not reference['requiresProductionRefusal'] and prediction['requiresProductionRefusal']:
        return 'invalid-converted-prediction'
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(256, 512), required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--compute-unit', choices=COMPUTE_POLICIES, default='cpuOnly')
    parser.add_argument('--ensemble', choices=('none', 'x8'), default='none')
    parser.add_argument('--case', action='append', choices=CASE_NAMES, dest='cases')
    parser.add_argument('--warm-predictions', type=int, choices=range(0, 6), default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError(f'Refuse to overwrite existing validation evidence: {args.report}')
    verify_official_sources()
    checkpoint_manifest()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import numpy as np
    import torch
    import coremltools as ct
    from PIL import Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    model_path = args.model or root / f'Targets/TruongNoiserApp/Models/CGNet_SIDD_{args.size}.mlmodelc'
    if not model_path.is_dir():
        raise RuntimeError(f'Missing real trained Core ML candidate: {model_path}')
    if model_path.suffix != '.mlmodelc':
        raise RuntimeError('Validate the compiled production candidate, not an uncompiled package')
    validate_compiled_tensor_contract(model_path, args.size)
    asset_digest, asset_files = compiled_identity(model_path)
    report = {
        'model': 'CGNet-SIDD', 'asset': str(model_path), 'size': args.size,
        'inputShape': [1, 3, args.size, args.size], 'outputShape': [1, 3, args.size, args.size],
        'tensorType': 'FLOAT16', 'architecture': ARCHITECTURE,
        'compiledDirectorySHA256': asset_digest, 'compiledFiles': asset_files,
        'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
        'reference': 'Unmodified official FP32 CascadedGaze + LayerNormFunction, GCE GELU, nn.Upsample nearest',
        'referenceInputs': 'Identical half-rounded developed-sRGB RGB tensors',
        'photoFixture': {
            'source': 'Pinned official NAFNet/demo/noisy.png, reused solely as a noisy-photo tensor fixture',
            'sha256': '403034182fa320130dae0d75b92e85e0850771378e674d65455c403a4958e29c',
            'largerShapeExtension': 'Repeat original pixels without interpolation; not a SIDD benchmark score',
        },
        'computeRequested': args.compute_unit, 'actualANEPlacement': 'NOT_MEASURED',
        'scope': 'Mac raw Core ML tensors; not a phone, SIDD score, HDR adapter or full-photo quality benchmark',
        'ensemble': args.ensemble, 'strengthsApplied': False, 'outputClamped': False,
        'fidelityGate': {'psnrMinDB': 50, 'maxAbsoluteErrorExclusive': .02},
        'productionRawRange': [-.25, 1.25],
        'cases': [], 'failures': [], 'finiteFidelityComparisons': 0,
        'refusalObservations': 0, 'warmPredictionSeconds': [], 'passed': False,
    }
    reference = model = None
    try:
        # This fresh official instance must never receive either conversion-only
        # lowering. The verifier intentionally does not import those functions.
        reference = load_official_model()
        report['parameters'] = sum(parameter.numel() for parameter in reference.parameters())
        policy = {
            'cpuOnly': ct.ComputeUnit.CPU_ONLY, 'cpuAndGPU': ct.ComputeUnit.CPU_AND_GPU,
            'cpuAndNeuralEngine': ct.ComputeUnit.CPU_AND_NE, 'all': ct.ComputeUnit.ALL,
        }[args.compute_unit]
        started = time.perf_counter()
        model_type = ct.models.CompiledMLModel if model_path.suffix == '.mlmodelc' else ct.models.MLModel
        model = model_type(str(model_path), compute_units=policy)
        report['modelLoadSeconds'] = time.perf_counter() - started
        transformations = [(turns, mirrored) for mirrored in (False, True) for turns in range(4)] if args.ensemble == 'x8' else [(0, False)]
        report['orientationOrder'] = [{'quarterTurnsCCW': turns, 'mirrorHorizontalBeforeRotation': mirrored}
                                      for turns, mirrored in transformations]
        all_cases = make_cases(args.size, np, Image)
        selected = args.cases or list(CASE_NAMES)
        expected_shape = (1, 3, args.size, args.size)
        for name in selected:
            value = np.ascontiguousarray(all_cases[name], dtype=np.float16)
            expected_sum = np.zeros(expected_shape, dtype=np.float32)
            actual_sum = np.zeros(expected_shape, dtype=np.float32)
            orientation_results = []
            unsafe_reference = unsafe_actual = invalid_output = False
            torch_seconds = prediction_seconds = 0.0
            for turns, mirrored in transformations:
                supplied = oriented(value, turns, mirrored, np)
                started = time.perf_counter()
                with torch.inference_mode():
                    expected = reference(torch.from_numpy(supplied.astype(np.float32))).numpy()
                torch_seconds += time.perf_counter() - started
                started = time.perf_counter()
                outputs = model.predict({'input_tile': supplied})
                prediction_seconds += time.perf_counter() - started
                if set(outputs) != {'output_tile'}:
                    raise RuntimeError(f'Unexpected converted output names: {sorted(outputs)}')
                actual = np.asarray(outputs['output_tile'])
                if expected.shape != expected_shape or actual.shape != expected_shape:
                    raise RuntimeError(f'Invalid trained prediction shapes: reference{expected.shape}, CoreML{actual.shape}')
                validate_prediction_bridge_dtype(actual, np)
                expected_state, actual_state = raw_summary(expected, np), raw_summary(actual, np)
                unsafe_reference |= expected_state['requiresProductionRefusal']
                unsafe_actual |= actual_state['requiresProductionRefusal']
                invalid_output |= not actual_state['finite']
                orientation_result = {
                    'pythonBridgeDtype': str(actual.dtype),
                    'quarterTurnsCCW': turns, 'mirrorHorizontalBeforeRotation': mirrored,
                    'reference': expected_state, 'prediction': actual_state,
                }
                refusal_failure = raw_refusal_failure(expected_state, actual_state)
                if refusal_failure:
                    report['failures'].append(f'{name}-orientation{turns}-mirror{mirrored}-{refusal_failure}')
                # Raw-invalid outputs are never used to create plausible averaged
                # pixels. The case records an explicit refusal observation below.
                if not expected_state['requiresProductionRefusal'] and not actual_state['requiresProductionRefusal']:
                    raw_difference = actual.astype(np.float64) - expected.astype(np.float64)
                    raw_mse = float(np.mean(raw_difference * raw_difference))
                    raw_psnr = 120.0 if raw_mse == 0 else -10 * math.log10(raw_mse)
                    raw_maximum = float(np.max(np.abs(raw_difference)))
                    orientation_result.update({'psnrDB': raw_psnr, 'maxAbsoluteError': raw_maximum})
                    if raw_psnr < 50 or raw_maximum >= .02:
                        report['failures'].append(f'{name}-orientation{turns}-mirror{mirrored}-fidelity')
                    expected_sum += undo_orientation(expected, turns, mirrored, np)
                    actual_sum += undo_orientation(actual.astype(np.float32), turns, mirrored, np)
                orientation_results.append(orientation_result)
            case = {'case': name, 'torchSeconds': torch_seconds, 'predictionSeconds': prediction_seconds,
                    'orientations': orientation_results}
            if unsafe_reference:
                case['status'] = 'REQUIRES_SAFE_REFUSAL'
                case['rawConvertedOutputRequiresRefusal'] = unsafe_actual
                report['refusalObservations'] += 1
                if not unsafe_actual:
                    report['failures'].append(name + '-masked-unsafe-reference')
            elif unsafe_actual or invalid_output:
                case['status'] = 'INVALID_CONVERTED_PREDICTION'
                report['failures'].append(name)
            else:
                expected = expected_sum / len(transformations)
                actual = actual_sum / len(transformations)
                difference = actual.astype(np.float64) - expected.astype(np.float64)
                mse = float(np.mean(difference * difference))
                psnr = 120.0 if mse == 0 else -10 * math.log10(mse)
                maximum = float(np.max(np.abs(difference)))
                case.update({
                    'status': 'PASS' if psnr >= 50 and maximum < .02 else 'FIDELITY_FAILED',
                    'psnrDB': psnr, 'maxAbsoluteError': maximum,
                    'minPrediction': float(actual.min()), 'maxPrediction': float(actual.max()),
                    'meanAbsoluteChangeFromInput': float(np.mean(np.abs(actual - value.astype(np.float32)))),
                })
                report['finiteFidelityComparisons'] += 1
                if case['status'] != 'PASS':
                    report['failures'].append(name)
            report['cases'].append(case)
            print(json.dumps(case, allow_nan=False), flush=True)
        demo = np.ascontiguousarray(all_cases['official-noisy-demo'], dtype=np.float16)
        for _ in range(args.warm_predictions):
            started = time.perf_counter()
            for turns, mirrored in transformations:
                output = np.asarray(model.predict({'input_tile': oriented(demo, turns, mirrored, np)})['output_tile'])
                if raw_summary(output, np)['requiresProductionRefusal']:
                    raise RuntimeError('Invalid raw warm prediction; safe refusal required')
            report['warmPredictionSeconds'].append(time.perf_counter() - started)
        report['passed'] = not report['failures'] and report['finiteFidelityComparisons'] > 0
        if report['finiteFidelityComparisons'] == 0:
            report['failures'].append('no-finite-fidelity-comparisons')
    except Exception as error:
        report['failures'].append(f'{type(error).__name__}: {error}')
        print(json.dumps({'error': report['failures'][-1]}), flush=True)
    finally:
        try:
            unchanged = compiled_identity(model_path) == (asset_digest, asset_files)
            report['compiledAssetUnchanged'] = unchanged
            if not unchanged:
                report['failures'].append('compiled-candidate-changed-during-validation')
                report['passed'] = False
        except Exception as error:
            report['compiledAssetUnchanged'] = False
            report['failures'].append(f'compiled-identity-check: {type(error).__name__}: {error}')
            report['passed'] = False
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        del model, reference
        gc.collect()
    print(json.dumps({key: report[key] for key in (
        'passed', 'finiteFidelityComparisons', 'refusalObservations', 'warmPredictionSeconds', 'failures',
    )}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
