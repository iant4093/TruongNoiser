#!/usr/bin/env python3
"""Convert one fixed official CGNet-SIDD shape in a bounded, serialized process.

The independent reference always uses a fresh unmodified upstream instance.
This script does not clamp predictions, alter trained weights, or promote an
unvalidated candidate into the app. --metadata-only imports no ML runtime.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

from cgnet_reference import (
    ARCHITECTURE, COMMIT, DEFAULT_CHECKPOINT, checkpoint_manifest,
    load_official_model, prepare_conversion_only_context, prepare_conversion_only_norms,
    prepare_conversion_only_downsamples,
    sha256, verify_official_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(256, 512), required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'build/ConvertedModels')
    parser.add_argument('--precision', choices=('stable-fp16', 'fp16', 'fp32'), default='fp32')
    parser.add_argument('--context-lowering', choices=('gather', 'matmul'), default='gather')
    parser.add_argument('--downsample-lowering', choices=('native', 'unshuffle'), default='unshuffle')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    verify_official_sources()
    checkpoint = checkpoint_manifest()
    if args.checkpoint.is_symlink() or args.checkpoint.stat().st_size != checkpoint['bytes'] or sha256(args.checkpoint) != checkpoint['sha256']:
        raise RuntimeError('Checkpoint hash/size mismatch before ML runtime import')
    destination = args.output_dir / f'CGNet_SIDD_{args.size}.mlpackage'
    if args.metadata_only:
        print(json.dumps({'commit': COMMIT, 'checkpointSHA256': checkpoint['sha256'],
                          'architecture': ARCHITECTURE, 'tileSize': args.size,
                          'contextLowering': args.context_lowering,
                          'downsampleLowering': args.downsample_lowering,
                          'input': 'gamma-sRGB-RGB-0to1, NCHW RGB',
                          'output': 'unclamped official native1x RGB prediction'}, indent=2))
        return
    if destination.exists():
        raise RuntimeError(f'Refuse to replace an existing candidate: {destination}')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import numpy as np
    import torch
    import coremltools as ct
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    network = load_official_model(args.checkpoint)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    norm_count = prepare_conversion_only_norms(network)
    context_count = prepare_conversion_only_context(network, args.size, mode=args.context_lowering)
    downsample_count = prepare_conversion_only_downsamples(network) if args.downsample_lowering == 'unshuffle' else 0
    print(f'Loaded strict official CGNet-SIDD: {parameters} parameters; {norm_count} LayerNorm and {context_count} exact {args.context_lowering} context lowerings', flush=True)

    class StaticShapeForward(torch.nn.Module):
        """Elide only no-op outer padding/cropping for fixed multiples of16.

        Official learned modules, GCE stride/padding, exact GELU and global pooling
        remain intact. Nearest interpolation uses exact fixed source indices in
        the conversion instance. Tile dependence needs separate seam validation.
        """
        def __init__(self, model):
            super().__init__()
            self.network = model

        def forward(self, inp):
            model = self.network
            x = model.intro(inp)
            encodings = []
            for encoder, down in zip(model.encoders, model.downs):
                x = encoder(x)
                encodings.append(x)
                x = down(x)
            x = model.middle_blks(x)
            for decoder, up, skip in zip(model.decoders, model.ups, encodings[::-1]):
                x = decoder(up(x) + skip)
            return model.ending(x) + inp

    model = StaticShapeForward(network).eval()
    example = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True, check_trace=False)
    trace_seconds = time.perf_counter() - started
    del network, model, example
    gc.collect()
    print(f'Traced official CGNet graph in {trace_seconds:.3f}s; converting', flush=True)
    conversion_start = time.perf_counter()
    if args.precision == 'stable-fp16':
        # Learned convolutions retain compact FP16 weights. Keep normalization, GELU,
        # gates and residual arithmetic in FP32: all-FP16 missed the per-orientation
        # fidelity gate on fine stripes with a CPU+NE request. Never hide that miss
        # by judging only the final average or lowering the reference threshold.
        sensitive = {'reduce_mean', 'pow', 'sqrt', 'real_div', 'sub', 'add', 'mul', 'gelu'}
        precision = ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in sensitive)
    else:
        precision = ct.precision.FLOAT16 if args.precision == 'fp16' else ct.precision.FLOAT32
    pipeline = ct.PassPipeline.DEFAULT
    # The default affine→BatchNorm rewrite inserts epsilon=1e-5 with variance=1,
    # changing trained gamma*x+beta into gamma*x/sqrt(1+epsilon)+beta. CGNet is
    # sensitive to that systematic perturbation. Its source has no BatchNorm.
    pipeline.remove_passes({'common::fuse_elementwise_to_batchnorm'})
    converted = ct.convert(
        traced, convert_to='mlprogram',
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name='input_tile', shape=(1, 3, args.size, args.size), dtype=np.float16)],
        outputs=[ct.TensorType(name='output_tile', dtype=np.float16)],
        skip_model_load=True,
        pass_pipeline=pipeline,
    )
    program = converted._mil_program
    def operations(block):
        for op in block.operations:
            yield op
            for nested in op.blocks:
                yield from operations(nested)
    graph_ops = [op for function in program.functions.values() for op in operations(function)]
    if any(op.op_type == 'batch_norm' for op in graph_ops):
        raise RuntimeError('A non-source BatchNorm appeared in the converted CGNet graph')
    converted.author = 'CascadedGaze authors; Core ML conversion by TruongNoiser'
    converted.license = 'MIT; derived NAFNet/BasicSR support notices are bundled with the app.'
    converted.short_description = f'Official trained CGNet SIDD, native1x {args.size}, developed sRGB RGB.'
    converted.input_description['input_tile'] = 'NCHW RGB gamma sRGB [0,1]. No alpha, Bayer or log-P3.'
    converted.output_description['output_tile'] = 'Native1x RGB, unclamped official prediction; alpha is outside the network.'
    metadata = {
        'officialRepository': 'https://github.com/Ascend-Research/CascadedGaze',
        'officialCommit': COMMIT, 'checkpointSHA256': checkpoint['sha256'],
        'inputDomain': 'gamma-sRGB-RGB-0to1', 'scaleFactor': '1',
        'computePrecision': args.precision,
        'contextLowering': args.context_lowering,
        'downsampleLowering': args.downsample_lowering,
        'reference': 'unmodified official FP32 CascadedGaze and LayerNormFunction',
    }
    for key, value in metadata.items():
        converted.user_defined_metadata[key] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted.save(str(destination))
    report = {
        'model': 'CGNet-SIDD', 'variant': destination.stem, 'tileSize': args.size,
        'scaleFactor': 1, 'inputName': 'input_tile', 'outputName': 'output_tile',
        'inputShape': [1, 3, args.size, args.size], 'outputShape': [1, 3, args.size, args.size],
        'inputDomain': 'gamma-sRGB-RGB-0to1', 'outputClamped': False, 'tensorType': 'FLOAT16',
        'computePrecision': args.precision, 'architecture': ARCHITECTURE, 'parameters': parameters,
        'officialCommit': COMMIT, 'checkpointSHA256': checkpoint['sha256'],
        'conversionOnlyLayerNormLoweringCount': norm_count,
        'disabledOptimizationPasses': ['common::fuse_elementwise_to_batchnorm'],
        'syntheticBatchNormCount': 0,
        'conversionOnlyExactNearestContextLoweringCount': context_count,
        'downsampleLowering': args.downsample_lowering,
        'conversionOnlyDownsampleLoweringCount': downsample_count,
        'contextLowering': args.context_lowering,
        'nearestInterpolationSourceIndex': 'floor(destinationIndex*sourceSize/targetSize)',
        'fixedShapeZeroPaddingAndFullCropElided': True,
        'officialGELURetained': True,
        'globalPoolingAndStridedContextRemainTileDependent': True,
        'traceSeconds': trace_seconds, 'conversionSeconds': time.perf_counter() - conversion_start,
        'versions': {'python': os.sys.version, 'torch': torch.__version__,
                     'numpy': np.__version__, 'coremltools': ct.__version__},
        'files': [{'path': str(path.relative_to(destination)), 'bytes': path.stat().st_size,
                   'sha256': sha256(path)} for path in sorted(destination.rglob('*')) if path.is_file()],
        'validation': 'PENDING independent official-reference parity, tiled seams, runtime and quality checks',
    }
    (args.output_dir / f'{destination.stem}.conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
