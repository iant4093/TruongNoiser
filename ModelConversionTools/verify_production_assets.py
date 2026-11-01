#!/usr/bin/env python3
"""Stream-check the four native denoiser assets, pinned references and actual app resources."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_cgnet_parity import validate_compiled_tensor_contract
from verify_cgnet_parity import compiled_identity, COMPUTE_POLICIES
from bundle_cgnet import validate_report

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'Targets/TruongNoiserApp/Models'
EXPECTED = {f'{family}_{size}' for family in ('NAFNet_w64', 'CGNet_SIDD') for size in (256, 512)}
PINS = {
    'NAFNet_w64': ('checkpoint-manifest.json', 'official-source-manifest.json',
        'cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445'),
    'CGNet_SIDD': ('cgnet-checkpoint-manifest.json', 'cgnet-official-source-manifest.json',
        'd7daaece4350f2e3d505155d1712d22929fec06b463532e076de217996a3462d'),
}


def sha256_file(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def directory_digest(path):
    result = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        result.update(file.relative_to(path).as_posix().encode() + b'\0')
        result.update(sha256_file(file).encode() + b'\n')
    return result.hexdigest()


def verify(models, checkpoints=None, built=None):
    inventory = json.loads((models / 'Provenance/model-inventory.json').read_text())
    errors = []
    entries = inventory.get('models', [])
    if inventory.get('canonicalModelCount') != 4 or len(entries) != 4 or {e['identifier'] for e in entries} != EXPECTED:
        errors.append('Expected exactly four NAFNet/CGNet native-resolution variants')
    expected_choices = {'NAFNet_w64': ('NAFNet_w64', 1), 'CGNet_SIDD': ('CGNet_SIDD', 1),
                        'CGNet_SIDD_x8': ('CGNet_SIDD', 8)}
    choices = {e['identifier']: (e['baseModel'], e['ensemblePassCount']) for e in inventory.get('choices', [])}
    if choices != expected_choices or len(inventory.get('choices', [])) != 3:
        errors.append('Expected three frozen modes with exactly one shared CGNet asset family')
    families = {e['identifier']: e for e in inventory.get('families', [])}
    if set(families) != set(PINS):
        errors.append('Missing trained model family provenance')
    for family, (checkpoint_name, source_name, pin) in PINS.items():
        checkpoint = json.loads((ROOT / 'ModelConversionTools' / checkpoint_name).read_text())
        source = json.loads((ROOT / 'ModelConversionTools' / source_name).read_text())
        if checkpoint['sha256'] != pin or families.get(family, {}).get('checkpointSHA256') != pin:
            errors.append(f'Trained checkpoint pin changed: {family}')
        for name in (checkpoint_name, source_name):
            bundled = models / 'Provenance' / name
            if not bundled.is_file() or sha256_file(bundled) != sha256_file(ROOT / 'ModelConversionTools' / name):
                errors.append(f'Missing/changed canonical provenance: {bundled}')
        for entry in source['files']:
            path = ROOT / 'ModelConversionTools' / entry['path']
            if not path.is_file() or path.is_symlink() or sha256_file(path) != entry['sha256']:
                errors.append(f'Official source digest drift: {path}')
        if checkpoints:
            path = checkpoints / Path(checkpoint['path']).name
            if not path.is_file() or path.is_symlink() or sha256_file(path) != pin:
                errors.append(f'Official trained checkpoint digest drift: {path}')
    for location in [models] + ([built / 'Models'] if built else []):
        if {p.stem for p in location.glob('*.mlmodelc')} != EXPECTED or list(location.glob('*.mlpackage')):
            errors.append(f'Unexpected model inventory: {location}')
        for entry in entries:
            identifier = entry['identifier']
            size = int(identifier.rsplit('_', 1)[1])
            family = identifier.rsplit('_', 1)[0]
            if (entry.get('inputShape') != [1, 3, size, size] or entry.get('outputShape') != [1, 3, size, size]
                    or entry.get('nativeScale') != 1 or entry.get('checkpointSHA256') != PINS[family][2]):
                errors.append(f'Invalid trained model contract: {identifier}')
            asset = location / entry['asset']
            if entry['asset'] != identifier + '.mlmodelc':
                errors.append(f'Noncanonical model asset path: {identifier}')
            try:
                identity = compiled_identity(asset)
                if (identity[0] != entry['assetDirectorySHA256']
                        or sum(item['bytes'] for item in identity[1]) != entry['bytes']):
                    raise RuntimeError('Trained asset digest/byte-count drift')
                validate_compiled_tensor_contract(asset, size)
                if family == 'CGNet_SIDD':
                    evidence = entry.get('parityEvidence', [])
                    expected_reports = {f'Provenance/cgnet-compiled-{size}-{policy}-{mode}.json'
                        for policy in COMPUTE_POLICIES for mode in ('none', 'x8')}
                    if len(evidence) != 8 or {item['path'] for item in evidence} != expected_reports:
                        raise RuntimeError('Incomplete model/policy/ensemble provenance')
                    by_path = {item['path']: item for item in evidence}
                    for policy in COMPUTE_POLICIES:
                        for mode in ('none', 'x8'):
                            item = by_path[f'Provenance/cgnet-compiled-{size}-{policy}-{mode}.json']
                            report_path = location / item['path']
                            if report_path.is_symlink() or sha256_file(report_path) != item['sha256']:
                                raise RuntimeError('Changed model parity evidence')
                            report = json.loads(report_path.read_text())
                            validate_report(report, size, policy, mode, identity)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                errors.append(f'Actual compiled tensor interface mismatch: {asset}: {error}')
        for family in families.values():
            notice = location / family['notice']
            if not notice.is_file() or sha256_file(notice) != family['licenseSHA256']:
                errors.append(f'Missing/changed official license: {notice}')
        for folder in ('Notices', 'Provenance'):
            if directory_digest(location / folder) != directory_digest(models / folder):
                errors.append(f'Missing/changed complete {folder} tree: {location}')
        for unexpected in list(location.rglob('*.pth')) + list(location.rglob('*.py')):
            errors.append(f'Development-only file bundled in app: {unexpected}')
    for error in errors:
        print('FAIL:', error)
    if not errors:
        print('PASS: four real native-resolution variants, three modes, official sources/notices/provenance' +
              (' and built app resources' if built else ''))
    return int(bool(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=MODELS)
    parser.add_argument('--checkpoints-dir', type=Path)
    parser.add_argument('--built-app', type=Path)
    args = parser.parse_args()
    raise SystemExit(verify(args.models_dir, args.checkpoints_dir, args.built_app))
