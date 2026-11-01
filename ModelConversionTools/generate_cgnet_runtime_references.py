#!/usr/bin/env python3
"""Retain independent official FP32 edge/interior samples for actual Swift/iOS predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from cgnet_reference import CHECKPOINT_SHA256, COMMIT, load_official_model, verify_official_sources
from verify_cgnet_parity import make_cases, compiled_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(256, 512), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse to replace an existing independent fixture')
    verify_official_sources()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = load_official_model()  # No conversion lowerings or Core ML predictions here.
    cases = make_cases(args.size, np, Image)
    report = {'schemaVersion': 1, 'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
        'size': args.size, 'compiledDirectorySHA256': compiled_identity(args.model)[0],
        'scope': 'Unmodified official FP32 oracle, half-rounded inputs; not phone image quality', 'cases': []}
    for name in ('gradient', 'fine-color-stripes'):
        for orientation in (0, 3, 5):
            supplied = cases[name]
            if orientation >= 4:
                supplied = np.flip(supplied, axis=3)
            supplied = np.ascontiguousarray(np.rot90(supplied, -(orientation % 4), axes=(2, 3)), dtype=np.float16)
            with torch.inference_mode():
                expected = reference(torch.from_numpy(supplied.astype(np.float32))).numpy()
            if not np.isfinite(expected).all() or not np.all((expected >= -.25) & (expected <= 1.25)):
                raise RuntimeError('A runtime fidelity fixture must be a valid raw prediction')
            positions = sorted({0, 1, args.size // 3, args.size // 2, args.size - 2, args.size - 1})
            indices = [channel * args.size * args.size + y * args.size + x
                for channel in range(3) for y in positions for x in positions]
            report['cases'].append({'name': name, 'orientation': orientation,
                'inputSHA256': hashlib.sha256(supplied.tobytes()).hexdigest(),
                'fullReferenceSHA256': hashlib.sha256(expected.tobytes()).hexdigest(),
                'samples': [{'index': index, 'value': float(expected.reshape(-1)[index])} for index in indices]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: official FP32 runtime references', args.size, len(report['cases']))


if __name__ == '__main__':
    main()
