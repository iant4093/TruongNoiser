#!/usr/bin/env python3
"""Independent official FP32 NAFNet versus Core ML tensor parity and local timings."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from nafnet_reference import COMMIT, CHECKPOINT_SHA256, load_official_model, verify_official_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(256, 512), required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--compute-unit', choices=('cpuOnly','cpuAndGPU','cpuAndNeuralEngine','all'), default='cpuOnly')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import numpy as np
    import torch
    import coremltools as ct
    from PIL import Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    verify_official_sources()
    root = Path(__file__).resolve().parents[1]
    model_path = args.model or root / f'Targets/TruongNoiserApp/Models/NAFNet_w64_{args.size}.mlmodelc'
    if not model_path.is_dir(): raise RuntimeError(f'Missing real trained model: {model_path}')
    reference = load_official_model()  # unmodified official forward + LayerNormFunction
    policy = {'cpuOnly': ct.ComputeUnit.CPU_ONLY, 'cpuAndGPU': ct.ComputeUnit.CPU_AND_GPU,
              'cpuAndNeuralEngine': ct.ComputeUnit.CPU_AND_NE, 'all': ct.ComputeUnit.ALL}[args.compute_unit]
    started = time.perf_counter()
    model_type = ct.models.CompiledMLModel if model_path.suffix == '.mlmodelc' else ct.models.MLModel
    model = model_type(str(model_path), compute_units=policy)
    load_seconds = time.perf_counter() - started
    size = args.size
    rng = np.random.default_rng(251028)
    demo = np.asarray(Image.open(root/'ModelConversionTools/official/NAFNet/demo/noisy.png').convert('RGB'), dtype=np.float32) / 255
    if demo.shape[:2] != (size,size):
        # Reflection extends the actual official noisy example; no synthetic smoothing/resizing.
        demo = np.tile(demo, (math.ceil(size/demo.shape[0]), math.ceil(size/demo.shape[1]),1))[:size,:size]
    demo = demo.transpose(2,0,1)[None].astype(np.float16)
    grid = (np.indices((size,size)).sum(axis=0)%2).astype(np.float16)
    gradient = np.broadcast_to(np.linspace(0,1,size,dtype=np.float16), (size,size))
    cases = {
        'black': np.zeros((1,3,size,size),np.float16),
        'white': np.ones((1,3,size,size),np.float16),
        'near-black': np.full((1,3,size,size),1/1024,np.float16),
        'gradient': np.stack([gradient, gradient.T, gradient*.5])[None],
        'high-frequency-fine-grain': np.stack([.48+.04*grid]*3)[None],
        'bounded-photo-noise': np.clip(.5+rng.normal(0,.025,(1,3,size,size)),0,1).astype(np.float16),
        'high-frequency-color': np.stack([grid,1-grid,grid*.5])[None],
        'seeded-noise': rng.uniform(0,1,(1,3,size,size)).astype(np.float16),
        'official-noisy-demo': demo,
    }
    results=[]
    failures=[]
    for name, value in cases.items():
        value = np.ascontiguousarray(value, dtype=np.float16)
        started = time.perf_counter()
        with torch.inference_mode():
            expected = reference(torch.from_numpy(value.astype(np.float32))).numpy()
        torch_seconds = time.perf_counter()-started
        started = time.perf_counter()
        prediction = np.asarray(model.predict({'input_tile': value})['output_tile'])
        prediction_seconds = time.perf_counter()-started
        if prediction.shape != expected.shape or prediction.shape != (1,3,size,size):
            raise RuntimeError(f'Invalid trained prediction shape: {prediction.shape}')
        reference_finite=bool(np.isfinite(expected).all())
        actual_finite=bool(np.isfinite(prediction).all())
        reference_safe=reference_finite and bool(np.all((expected>=-.25)&(expected<=1.25)))
        actual_safe=actual_finite and bool(np.all((prediction>=-.25)&(prediction<=1.25)))
        if not reference_safe:
            # The official polynomial network is unstable on these off-distribution inputs.
            # A raw converted output must trigger the same production refusal, never a clamp
            # or plausible-looking identity substitute. This is a safety test, not image quality.
            case={'case':name,'referenceFinite':reference_finite,'predictionFinite':actual_finite,
                  'referenceMin':float(np.nanmin(expected)),'referenceMax':float(np.nanmax(expected)),
                  'predictionSeconds':prediction_seconds,'status':'REQUIRES_SAFE_REFUSAL',
                  'rawConvertedOutputRequiresRefusal':not actual_safe}
            results.append(case)
            if actual_safe: failures.append(name+'-masked-unsafe-reference')
            print(json.dumps(case),flush=True)
            continue
        if not actual_finite:
            results.append({'case':name,'status':'INVALID_CONVERTED_PREDICTION','referenceFinite':True})
            failures.append(name)
            continue
        difference = prediction.astype(np.float64)-expected.astype(np.float64)
        mse=float(np.mean(difference*difference))
        psnr=120.0 if mse==0 else -10*math.log10(mse)
        maximum=float(np.max(np.abs(difference)))
        case={'case':name,'psnrDB':psnr,'maxAbsoluteError':maximum,
              'torchSeconds':torch_seconds,'predictionSeconds':prediction_seconds,
              'minPrediction':float(prediction.min()),'maxPrediction':float(prediction.max()),
              'meanAbsoluteChangeFromInput':float(np.mean(np.abs(prediction.astype(np.float32)-value.astype(np.float32))))}
        results.append(case)
        print(json.dumps(case), flush=True)
        if psnr<50 or maximum>=.02: failures.append(name)
    warm=[]
    for _ in range(3):
        started=time.perf_counter()
        prediction=model.predict({'input_tile':demo})['output_tile']
        warm.append(time.perf_counter()-started)
        if not np.isfinite(prediction).all(): raise RuntimeError('Non-finite warm prediction')
    report={'model':'NAFNet-SIDD-width64','asset':str(model_path),'size':size,'officialCommit':COMMIT,
            'checkpointSHA256':CHECKPOINT_SHA256,'reference':'Unmodified official FP32 NAFNet + LayerNormFunction',
            'computeRequested':args.compute_unit,'actualANEPlacement':'NOT_MEASURED',
            'scope':'Mac Core ML tensor predictions; not a phone or full-image pipeline benchmark',
            'modelLoadSeconds':load_seconds,'warmPredictionSeconds':warm,'cases':results,
            'fidelityGate':{'psnrMinDB':50,'maxAbsoluteErrorExclusive':.02},'failures':failures,
            'passed':not failures}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':not failures,'loadSeconds':load_seconds,'warmSeconds':warm,'failures':failures}),flush=True)
    del model,reference,cases
    gc.collect()
    return 1 if failures else 0

if __name__=='__main__': raise SystemExit(main())
