"""Load the pinned, byte-unmodified official NAFNet SIDD reference lazily.

The package loader avoids BasicSR's unrelated training/plugin registry imports;
the network, LayerNorm autograd implementation, local support and logger remain official.
No Python ML package is imported merely by importing this file or requesting CLI help.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official' / 'NAFNet'
COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'
CHECKPOINT_SHA256 = 'cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445'
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'NAFNet-SIDD-width64.pth'
ARCHITECTURE = {'img_channel': 3, 'width': 64, 'enc_blk_nums': [2, 2, 4, 8],
                'middle_blk_num': 12, 'dec_blk_nums': [2, 2, 2, 2]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_sources() -> dict:
    manifest = json.loads((ROOT / 'official-source-manifest.json').read_text())
    if manifest['commit'] != COMMIT:
        raise RuntimeError('Official source commit mismatch')
    for item in manifest['files']:
        path = ROOT / item['path']
        if path.is_symlink() or not path.is_file() or sha256(path) != item['sha256']:
            raise RuntimeError(f'Pinned official source changed: {path}')
    return manifest


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, OFFICIAL / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load official module {name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_network_class():
    verify_official_sources()
    loaded = sys.modules.get('basicsr.models.archs.NAFNet_arch')
    if loaded is not None:
        if Path(loaded.__file__).resolve() != (OFFICIAL / 'basicsr/models/archs/NAFNet_arch.py').resolve():
            raise RuntimeError('Another BasicSR/NAFNet implementation is already imported')
        return loaded.NAFNet
    for name, relative in [('basicsr', 'basicsr'), ('basicsr.models', 'basicsr/models'),
                           ('basicsr.models.archs', 'basicsr/models/archs'),
                           ('basicsr.utils', 'basicsr/utils')]:
        if name in sys.modules:
            raise RuntimeError(f'Refuse to replace an already imported package {name}')
        module = types.ModuleType(name)
        module.__path__ = [str(OFFICIAL / relative)]
        sys.modules[name] = module
    _load_module('basicsr.utils.dist_util', 'basicsr/utils/dist_util.py')
    logger = _load_module('basicsr.utils.logger', 'basicsr/utils/logger.py')
    sys.modules['basicsr.utils'].get_root_logger = logger.get_root_logger
    _load_module('basicsr.models.archs.arch_util', 'basicsr/models/archs/arch_util.py')
    _load_module('basicsr.models.archs.local_arch', 'basicsr/models/archs/local_arch.py')
    return _load_module('basicsr.models.archs.NAFNet_arch', 'basicsr/models/archs/NAFNet_arch.py').NAFNet


def load_official_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    import torch
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_symlink() or sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint must match the pinned official SIDD-width64 SHA-256')
    payload = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('params'), dict):
        raise RuntimeError('Expected the official BasicSR checkpoint params state dictionary')
    weights = payload['params']
    if not weights or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()):
        raise RuntimeError('Official checkpoint contains invalid state entries')
    model = official_network_class()(**ARCHITECTURE)
    result = model.load_state_dict(weights, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Strict official trained weight load failed: {result}')
    model.eval().requires_grad_(False)
    return model


def prepare_conversion_only_norms(model) -> int:
    """Lower only custom-autograd LayerNorm to identical official forward primitives.

    Mutates the conversion instance only. Reference predictions always load another instance
    executing the original official LayerNormFunction.apply implementation.
    """
    def official_forward_primitives(self, x):
        channels = x.shape[1]
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + self.eps).sqrt()
        return self.weight.view(1, channels, 1, 1) * normalized + self.bias.view(1, channels, 1, 1)

    count = 0
    for module in model.modules():
        if type(module).__name__ == 'LayerNorm2d':
            module.forward = types.MethodType(official_forward_primitives, module)
            count += 1
    if count != 72:
        raise RuntimeError(f'Unexpected official width64 LayerNorm inventory: {count}')
    return count
