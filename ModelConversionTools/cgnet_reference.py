"""Load a byte-pinned, unmodified official CascadedGaze SIDD reference lazily.

Only the required official BasicSR modules are loaded. Importing this module does
not import torch/coremltools or execute a network. Conversion-only substitutions
must be applied to another instance, never to the reference used for parity.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official' / 'CascadedGaze'
COMMIT = '0daa98442b928274ff50a1b5e594a2b53c94879d'
CHECKPOINT_SHA256 = 'd7daaece4350f2e3d505155d1712d22929fec06b463532e076de217996a3462d'
CHECKPOINT_BYTES = 477119009
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'CGNet-SIDD.pth'
ARCHITECTURE = {
    'img_channel': 3, 'width': 60, 'enc_blk_nums': [2, 2, 4, 6],
    'middle_blk_num': 10, 'dec_blk_nums': [2, 2, 2, 2],
    'GCE_CONVS_nums': [3, 3, 2, 2],
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_official_sources() -> dict:
    manifest = json.loads((ROOT / 'cgnet-official-source-manifest.json').read_text())
    if manifest['commit'] != COMMIT:
        raise RuntimeError('CascadedGaze official source commit mismatch')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.resolve().is_relative_to(OFFICIAL.resolve()):
            raise RuntimeError('Official source manifest contains an unexpected path')
        if path.is_symlink() or not path.is_file() or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Pinned CascadedGaze source changed: {path}')
    return manifest


def checkpoint_manifest() -> dict:
    manifest = json.loads((ROOT / 'cgnet-checkpoint-manifest.json').read_text())
    if (manifest['officialReferenceCommit'] != COMMIT or manifest['model'] != 'CGNet-SIDD'
            or manifest['sha256'] != CHECKPOINT_SHA256 or manifest['bytes'] != CHECKPOINT_BYTES):
        raise RuntimeError('Unexpected CGNet trained checkpoint provenance')
    return manifest


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, OFFICIAL / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load official CascadedGaze module: {name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_network_class():
    verify_official_sources()
    loaded = sys.modules.get('basicsr.models.archs.CGNet_arch')
    if loaded is not None:
        if Path(loaded.__file__).resolve() != (OFFICIAL / 'basicsr/models/archs/CGNet_arch.py').resolve():
            raise RuntimeError('Another CascadedGaze implementation is already imported')
        return loaded.CascadedGaze
    # NAFNet and CGNet carry distinct pinned BasicSR support. Run each loader in
    # its own process; never silently replace an imported implementation.
    for name, relative in [
        ('basicsr', 'basicsr'), ('basicsr.models', 'basicsr/models'),
        ('basicsr.models.archs', 'basicsr/models/archs'), ('basicsr.utils', 'basicsr/utils'),
    ]:
        if name in sys.modules:
            raise RuntimeError(f'Refuse to replace an already imported package: {name}')
        module = types.ModuleType(name)
        module.__path__ = [str(OFFICIAL / relative)]
        sys.modules[name] = module
    _load_module('basicsr.utils.dist_util', 'basicsr/utils/dist_util.py')
    logger = _load_module('basicsr.utils.logger', 'basicsr/utils/logger.py')
    sys.modules['basicsr.utils'].get_root_logger = logger.get_root_logger
    _load_module('basicsr.models.archs.arch_util', 'basicsr/models/archs/arch_util.py')
    _load_module('basicsr.models.archs.local_arch', 'basicsr/models/archs/local_arch.py')
    return _load_module('basicsr.models.archs.CGNet_arch', 'basicsr/models/archs/CGNet_arch.py').CascadedGaze


def load_official_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    import torch
    path = Path(checkpoint_path)
    manifest = checkpoint_manifest()
    if path.is_symlink() or path.stat().st_size != manifest['bytes'] or sha256(path) != manifest['sha256']:
        raise RuntimeError('Checkpoint must match the pinned official CGNet-SIDD bytes and SHA-256')
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('params'), dict):
        raise RuntimeError('Expected the official BasicSR checkpoint params state dictionary')
    weights = payload['params']
    if not weights or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()):
        raise RuntimeError('Official CGNet checkpoint contains invalid state entries')
    model = official_network_class()(**ARCHITECTURE)
    result = model.load_state_dict(weights, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Strict trained CGNet weight load failed: {result}')
    model.eval().requires_grad_(False)
    return model


def prepare_conversion_only_norms(model) -> int:
    """Lower custom-autograd LayerNorm only in a separate conversion instance."""
    def forward_primitives(self, x):
        channels = x.shape[1]
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + self.eps).sqrt()
        return self.weight.view(1, channels, 1, 1) * normalized + self.bias.view(1, channels, 1, 1)

    count = 0
    for module in model.modules():
        if type(module).__name__ == 'LayerNorm2d':
            module.forward = types.MethodType(forward_primitives, module)
            count += 1
    if count != 64:
        raise RuntimeError(f'Unexpected official CGNet LayerNorm inventory: {count}')
    return count


def prepare_conversion_only_downsamples(model) -> int:
    """Lower learned 2x2/stride2 convolutions to space-to-depth plus 1x1.

    Every (channel,dy,dx) coefficient retains its original association. Only the
    conversion instance changes; the official reference keeps strided convolutions.
    """
    import torch.nn.functional as functional

    def forward_unshuffle(self, value):
        separated = functional.pixel_unshuffle(value, 2)
        weights = self.weight.reshape(self.out_channels, self.in_channels * 4, 1, 1)
        return functional.conv2d(separated, weights, self.bias)

    if len(model.downs) != 4:
        raise RuntimeError('Unexpected official CGNet downsample inventory')
    for convolution in model.downs:
        if (convolution.kernel_size != (2, 2) or convolution.stride != (2, 2)
                or convolution.padding != (0, 0) or convolution.dilation != (1, 1)
                or convolution.groups != 1):
            raise RuntimeError('Unexpected official CGNet downsample geometry')
    for convolution in model.downs:
        convolution.forward = types.MethodType(forward_unshuffle, convolution)
    return 4


def prepare_conversion_only_context(model, size: int, mode: str = 'gather') -> int:
    """Lower fixed-shape nearest resizing to the exact official source indices.

    The upstream block constructs an Upsample module during every forward. Core
    ML's generic nearest-resize conventions need not match PyTorch for uneven
    ratios. Fixed index_select operations use floor(dst * source / target), the
    official PyTorch nearest mapping. All target dimensions here are powers of2,
    so these ratios are exactly representable; no bilinear approximation occurs.
    An optional separable0/1 matrix mapping computes M@context@M.T with the
    identical source indices. It provides a distinct backend lowering when a
    gather/concat specialization disagrees across compute policies. No learned
    convolution weights change, and neither mode patches the official reference.
    """
    import torch
    import torch.nn.functional as functional

    if size not in (256, 512):
        raise RuntimeError('CGNet conversion requires a validated fixed256/512 shape')
    if mode not in ('gather', 'matmul'):
        raise RuntimeError(f'Unsupported exact CGNet context lowering: {mode}')

    def forward_with_exact_nearest(self, inp):
        x = functional.gelu(self.conv2(self.conv1(self.norm1(inp))))
        first, second = x.chunk(2, dim=1)
        contexts = self.GCE(first + second)
        resized = []
        for index, context in enumerate(contexts):
            if self._conversion_context_lowering == 'matmul':
                mapping = getattr(self, f'_conversion_nearest_mapping_{index}')
                resized.append(torch.matmul(torch.matmul(mapping, context), mapping.transpose(0, 1)))
            else:
                indices = getattr(self, f'_conversion_nearest_indices_{index}')
                resized.append(context.index_select(2, indices).index_select(3, indices))
        x = torch.cat([x] + resized, dim=1)
        x = self.project_out(self.sca(x) * x)
        y = inp + self.dropout1(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + self.dropout2(x) * self.gamma

    count = 0
    for stage, encoder in enumerate(model.encoders):
        target = size // (2 ** stage)
        for block in encoder:
            if type(block).__name__ != 'CascadedGazeBlock':
                raise RuntimeError('Unexpected CGNet encoder block type')
            block._conversion_context_lowering = mode
            source = target
            for index, convolution in enumerate(block.GCE.depthwise_separable_convs):
                depthwise = convolution.depthwise
                kernel, stride, padding, dilation = (
                    depthwise.kernel_size[0], depthwise.stride[0],
                    depthwise.padding[0], depthwise.dilation[0],
                )
                if any(getattr(depthwise, key)[0] != getattr(depthwise, key)[1]
                       for key in ('kernel_size', 'stride', 'padding', 'dilation')):
                    raise RuntimeError('Unexpected asymmetric official CGNet geometry')
                source = (source + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
                if source <= 0:
                    raise RuntimeError('CGNet context branch cannot support this shape')
                indices = torch.tensor([(position * source) // target for position in range(target)],
                                       dtype=torch.int64)
                if mode == 'matmul':
                    mapping = torch.zeros((target, source), dtype=torch.float32)
                    mapping[torch.arange(target), indices] = 1
                    block.register_buffer(f'_conversion_nearest_mapping_{index}', mapping, persistent=False)
                else:
                    block.register_buffer(f'_conversion_nearest_indices_{index}', indices, persistent=False)
            if len(block.GCE.depthwise_separable_convs) != block.GCE_Conv:
                raise RuntimeError('Unexpected CGNet context branch count')
            block.forward = types.MethodType(forward_with_exact_nearest, block)
            count += 1
    if count != 14:
        raise RuntimeError(f'Unexpected official CGNet context block inventory: {count}')
    return count
