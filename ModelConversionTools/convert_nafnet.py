#!/usr/bin/env python3
"""Convert pinned official NAFNet-width64 SIDD, native 1x developed sRGB.

Run one fixed shape per bounded process. Never initializes a custom network, substitutes
weights, adds an output clamp, or imports the ML runtime for --help/metadata checks.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
from nafnet_reference import (ARCHITECTURE, CHECKPOINT_SHA256, COMMIT, DEFAULT_CHECKPOINT,
                              load_official_model, prepare_conversion_only_norms,
                              sha256, verify_official_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(256, 512), required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'build/ConvertedModels')
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    sources = verify_official_sources()
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint hash mismatch before ML runtime import')
    if args.metadata_only:
        print(json.dumps({'commit': sources['commit'], 'checkpointSHA256': CHECKPOINT_SHA256,
                          'tileSize': args.size, 'input': 'developed gamma sRGB [0,1], NCHW RGB',
                          'output': 'unclamped official native1x RGB prediction'}, indent=2))
        return
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import numpy as np
    import torch
    import coremltools as ct
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    model = load_official_model(args.checkpoint)
    parameters = sum(p.numel() for p in model.parameters())
    norm_count = prepare_conversion_only_norms(model)
    print(f'Loaded strict official trained model: {parameters} parameters; {norm_count} conversion-only LayerNorm lowerings', flush=True)
    class StaticShapeForward(torch.nn.Module):
        """Elide official zero padding/full-image cropping only for fixed divisor-aligned inputs.

        All learned modules/weights remain the official model's modules. Independent
        parity loads an unmodified official instance, including its padding/cropping.
        """
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, inp):
            network = self.network
            x = network.intro(inp)
            encodings = []
            for encoder, down in zip(network.encoders, network.downs):
                x = encoder(x)
                encodings.append(x)
                x = down(x)
            x = network.middle_blks(x)
            for decoder, up, skip in zip(network.decoders, network.ups, encodings[::-1]):
                x = decoder(up(x) + skip)
            return network.ending(x) + inp
    model = StaticShapeForward(model).eval()
    example = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True, check_trace=False)
    trace_seconds = time.perf_counter() - started
    del model, example
    gc.collect()
    print(f'Official graph traced in {trace_seconds:.3f}s; converting', flush=True)
    conversion_start = time.perf_counter()
    precision = ct.precision.FLOAT16 if args.precision == 'fp16' else ct.precision.FLOAT32
    converted = ct.convert(
        traced, convert_to='mlprogram', compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name='input_tile', shape=(1, 3, args.size, args.size), dtype=np.float16)],
        outputs=[ct.TensorType(name='output_tile', dtype=np.float16)],
        skip_model_load=True,
    )
    converted.author = 'Megvii NAFNet authors; Core ML conversion by TruongNoiser'
    converted.license = 'MIT; BasicSR support Apache-2.0. See bundled NAFNet-LICENSE.txt.'
    converted.short_description = f'Official trained NAFNet SIDD width64, native1x {args.size}, developed sRGB RGB.'
    converted.input_description['input_tile'] = 'NCHW RGB gamma sRGB [0,1]. No alpha or log-P3.'
    converted.output_description['output_tile'] = 'Native1x RGB, unclamped official network output. Alpha is outside the network.'
    for key, value in {'officialRepository': 'https://github.com/megvii-research/NAFNet',
                       'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
                       'inputDomain': 'gamma-sRGB-RGB-0to1', 'scaleFactor': '1',
                       'computePrecision': args.precision,
                       'reference': 'unmodified official FP32 NAFNet and LayerNormFunction'}.items():
        converted.user_defined_metadata[key] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f'NAFNet_w64_{args.size}.mlpackage'
    if destination.exists():
        raise RuntimeError(f'Refuse to replace an existing model package: {destination}')
    converted.save(str(destination))
    model_files = [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                   for p in sorted(destination.rglob('*')) if p.is_file()]
    report = {'model': 'NAFNet-SIDD-width64', 'variant': destination.stem, 'tileSize': args.size,
              'scaleFactor': 1, 'inputName': 'input_tile', 'outputName': 'output_tile',
              'inputShape': [1, 3, args.size, args.size], 'outputShape': [1, 3, args.size, args.size],
              'inputDomain': 'gamma-sRGB-RGB-0to1', 'outputClamped': False, 'tensorType': 'FLOAT16',
              'computePrecision': args.precision, 'architecture': ARCHITECTURE, 'parameters': parameters,
              'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
              'conversionOnlyLayerNormLoweringCount': norm_count,
              'fixedShapeZeroPaddingAndFullCropElided': True,
              'traceSeconds': trace_seconds, 'conversionSeconds': time.perf_counter() - conversion_start,
              'versions': {'python': os.sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'coremltools': ct.__version__},
              'files': model_files, 'validation': 'PENDING independent official-reference parity and runtime measurement'}
    report_path = args.output_dir / f'{destination.stem}.conversion.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
