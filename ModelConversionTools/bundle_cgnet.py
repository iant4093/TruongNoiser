#!/usr/bin/env python3
"""Promote only complete, hash-bound CGNet evidence; retain the existing NAFNet pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from cgnet_reference import CHECKPOINT_SHA256, COMMIT, checkpoint_manifest, sha256, verify_official_sources
from verify_cgnet_parity import CASE_NAMES, COMPUTE_POLICIES, compiled_identity, validate_compiled_tensor_contract

ROOT = Path(__file__).resolve().parents[1]


def validate_report(report, size, policy, mode, identity):
    """Recheck raw orientations, rather than trusting an aggregate passed flag."""
    expected = {
        'model': 'CGNet-SIDD', 'size': size, 'computeRequested': policy, 'ensemble': mode,
        'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
        'inputShape': [1, 3, size, size], 'outputShape': [1, 3, size, size],
        'tensorType': 'FLOAT16', 'passed': True, 'failures': [], 'compiledAssetUnchanged': True,
        'compiledDirectorySHA256': identity[0], 'compiledFiles': identity[1],
        'strengthsApplied': False, 'outputClamped': False,
        'fidelityGate': {'psnrMinDB': 50, 'maxAbsoluteErrorExclusive': .02},
        'productionRawRange': [-.25, 1.25],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Failed, stale or incompatible CGNet report: {size}/{policy}/{mode}')
    for key in ('passed', 'compiledAssetUnchanged', 'strengthsApplied', 'outputClamped'):
        if type(report.get(key)) is not bool:
            raise RuntimeError('Report state must use explicit boolean values')
    cases = report.get('cases', [])
    if len(cases) != len(CASE_NAMES) or {c.get('case') for c in cases} != set(CASE_NAMES):
        raise RuntimeError('Complete high-frequency/photo case coverage is required')
    orientations = {(turn, mirror) for mirror in (False, True) for turn in range(4)} if mode == 'x8' else {(0, False)}
    finite_count = refusal_count = 0
    for case in cases:
        rows = case.get('orientations', [])
        if len(rows) != len(orientations) or {(r.get('quarterTurnsCCW'), r.get('mirrorHorizontalBeforeRotation')) for r in rows} != orientations:
            raise RuntimeError('Missing or repeated raw ensemble orientation')
        unsafe_reference = False
        for row in rows:
            reference, prediction = row.get('reference', {}), row.get('prediction', {})
            validate_raw_summary(reference)
            validate_raw_summary(prediction)
            if reference.get('requiresProductionRefusal') is True:
                unsafe_reference = True
                if prediction.get('requiresProductionRefusal') is not True:
                    raise RuntimeError('Unsafe reference was hidden by a plausible converted prediction')
            else:
                if reference.get('finite') is not True or prediction.get('finite') is not True or prediction.get('requiresProductionRefusal') is not False:
                    raise RuntimeError('Invalid ordinary-photo prediction')
                validate_metrics(row)
        if unsafe_reference:
            refusal_count += 1
            if case.get('status') != 'REQUIRES_SAFE_REFUSAL' or case.get('rawConvertedOutputRequiresRefusal') is not True:
                raise RuntimeError('An off-domain refusal must remain explicit')
        else:
            finite_count += 1
            if case.get('status') != 'PASS':
                raise RuntimeError('A complete photo-domain case failed')
            validate_metrics(case)
    if finite_count != report.get('finiteFidelityComparisons') or refusal_count != report.get('refusalObservations') or not finite_count:
        raise RuntimeError('Inconsistent reference comparison counts')
    warm = report.get('warmPredictionSeconds', [])
    if not warm or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in warm):
        raise RuntimeError('Actual warm predictions and timings are required')


def validate_metrics(row):
    psnr, maximum = row.get('psnrDB'), row.get('maxAbsoluteError')
    if (type(psnr) not in (float, int) or type(maximum) not in (float, int)
            or not math.isfinite(psnr) or not math.isfinite(maximum)
            or psnr < 50 or not 0 <= maximum < .02):
        raise RuntimeError('Raw/final reference fidelity gate failed')


def validate_raw_summary(summary):
    """Reject a claimed safe flag that disagrees with finite values or extrema."""
    finite, refusal = summary.get('finite'), summary.get('requiresProductionRefusal')
    if type(finite) is not bool or type(refusal) is not bool:
        raise RuntimeError('Raw prediction state requires explicit finite/refusal booleans')
    if 'minFinite' not in summary or 'maxFinite' not in summary:
        raise RuntimeError('Raw state must record both finite extrema explicitly')
    minimum, maximum = summary['minFinite'], summary['maxFinite']
    if minimum is None and maximum is None:
        if finite or not refusal:
            raise RuntimeError('An all-nonfinite tensor requires explicit refusal')
        return
    if (type(minimum) not in (float, int) or type(maximum) not in (float, int)
            or not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum):
        raise RuntimeError('Invalid raw finite extrema')
    expected_refusal = not finite or minimum < -.25 or maximum > 1.25
    if refusal is not expected_refusal:
        raise RuntimeError('Raw extrema/finite state disagree with the production refusal flag')


def load_json_snapshot(path):
    if path.is_symlink():
        raise RuntimeError(f'Provenance must not be a symbolic link: {path}')
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def assert_file_unchanged(path, digest):
    if path.is_symlink() or not path.is_file() or sha256(path) != digest:
        raise RuntimeError(f'Preflight file changed during promotion: {path}')


def stage_frozen_copy(source, destination, digest):
    """The copied bytes must equal the validated snapshot, not a newly changed source."""
    assert_file_unchanged(source, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert_file_unchanged(source, digest)
    assert_file_unchanged(destination, digest)


def assert_compiled_unchanged(assets):
    for asset, identity in assets:
        if compiled_identity(asset) != identity:
            raise RuntimeError(f'Compiled model changed during promotion: {asset}')


def same_or_absent(destination, source):
    if destination.is_symlink() or (destination.exists() and sha256(destination) != sha256(source)):
        raise RuntimeError(f'Refuse to replace different existing provenance: {destination}')


def promote(args):
    verify_official_sources()
    checkpoint = checkpoint_manifest()
    checkpoint_path = ROOT / 'ModelConversionTools' / checkpoint['path']
    if checkpoint_path.stat().st_size != checkpoint['bytes'] or sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError('Official checkpoint identity changed')
    models = args.models_dir
    inventory_path = models / 'Provenance/model-inventory.json'
    old, old_inventory_digest = load_json_snapshot(inventory_path)
    naf = [entry for entry in old['models'] if entry['identifier'].startswith('NAFNet_w64_')]
    if {entry['identifier'] for entry in naf} != {'NAFNet_w64_256', 'NAFNet_w64_512'}:
        raise RuntimeError('The existing validated NAFNet pair is required')
    naf_assets = []
    for entry in naf:
        identity = compiled_identity(models / entry['asset'])
        if identity[0] != entry['assetDirectorySHA256']:
            raise RuntimeError('Existing NAFNet asset changed')
        naf_assets.append((models / entry['asset'], identity))
    entries = list(naf)
    copies = []
    assets = []
    package_files = []
    for size in (256, 512):
        name = f'CGNet_SIDD_{size}'
        asset = args.compiled_root / (name + '.mlmodelc')
        validate_compiled_tensor_contract(asset, size)
        identity = compiled_identity(asset)
        destination = models / asset.name
        if destination.exists() and compiled_identity(destination) != identity:
            raise RuntimeError('Refuse to replace a different existing CGNet asset')
        conversion_path = args.converted_root / (name + '.conversion.json')
        conversion, conversion_digest = load_json_snapshot(conversion_path)
        if (conversion.get('checkpointSHA256') != CHECKPOINT_SHA256 or conversion.get('officialCommit') != COMMIT
                or conversion.get('tileSize') != size or conversion.get('outputClamped') is not False
                or conversion.get('computePrecision') != 'fp32' or conversion.get('downsampleLowering') != 'unshuffle'
                or conversion.get('conversionOnlyDownsampleLoweringCount') != 4):
            raise RuntimeError('Unexpected conversion recipe or source identity')
        package = args.converted_root / (name + '.mlpackage')
        for entry in conversion['files']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe converted package manifest')
            path = package / relative
            if path.is_symlink() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                raise RuntimeError('Converted source package changed after conversion')
            package_files.append((path, entry['sha256']))
        reports = []
        for policy in COMPUTE_POLICIES:
            for mode in ('none', 'x8'):
                filename = f'cgnet-compiled-{size}-{policy}-{mode}.json'
                source = args.validation_dir / filename
                report, report_digest = load_json_snapshot(source)
                validate_report(report, size, policy, mode, identity)
                copies.append((source, Path('Provenance') / filename, report_digest))
                reports.append({'path': 'Provenance/' + filename, 'sha256': report_digest})
        copies.append((conversion_path, Path('Provenance') / conversion_path.name, conversion_digest))
        assets.append((asset, identity))
        entries.append({'identifier': name, 'asset': asset.name, 'inputShape': [1, 3, size, size],
            'outputShape': [1, 3, size, size], 'nativeScale': 1, 'checkpointSHA256': CHECKPOINT_SHA256,
            'assetDirectorySHA256': identity[0], 'bytes': sum(f['bytes'] for f in identity[1]),
            'parityEvidence': reports})
    for filename in ('cgnet-checkpoint-manifest.json', 'cgnet-official-source-manifest.json'):
        source = ROOT / 'ModelConversionTools' / filename
        copies.append((source, Path('Provenance') / filename, sha256(source)))
    license_path = ROOT / 'ModelConversionTools/official/CascadedGaze/LICENSE.txt'
    copies.append((license_path, Path('Notices/CGNet-LICENSE.txt'), sha256(license_path)))
    for source, relative, digest in copies:
        assert_file_unchanged(source, digest)
        same_or_absent(models / relative, source)
    naf_notice = models / 'Notices/NAFNet-LICENSE.txt'
    naf_notice_digest = sha256(naf_notice)
    naf_manifest = json.loads((ROOT / 'ModelConversionTools/checkpoint-manifest.json').read_text())
    inventory = {'schemaVersion': 2, 'canonicalModelCount': 4, 'models': entries,
        'modelDomain': 'developed gamma sRGB; alpha outside model',
        'families': [
            {'identifier': 'NAFNet_w64', 'checkpointSHA256': naf_manifest['sha256'],
                'notice': 'Notices/NAFNet-LICENSE.txt', 'licenseSHA256': naf_notice_digest},
            {'identifier': 'CGNet_SIDD', 'checkpointSHA256': CHECKPOINT_SHA256,
                'notice': 'Notices/CGNet-LICENSE.txt', 'licenseSHA256': sha256(license_path)}],
        'choices': [{'identifier': name, 'baseModel': base, 'ensemblePassCount': passes}
            for name, base, passes in [('NAFNet_w64', 'NAFNet_w64', 1), ('CGNet_SIDD', 'CGNet_SIDD', 1),
                ('CGNet_SIDD_x8', 'CGNet_SIDD', 8)]]}
    # All evidence is checked before touching canonical assets. Stage and rehash the copies;
    # publish the inventory last, so interrupted promotion can be repeated only byte-identically.
    with tempfile.TemporaryDirectory(prefix='.cgnet-promotion-', dir=models.parent) as temporary:
        stage = Path(temporary)
        for asset, identity in assets:
            shutil.copytree(asset, stage / asset.name)
            if compiled_identity(stage / asset.name) != identity:
                raise RuntimeError('Staged compiled asset copy changed')
        for source, relative, digest in copies:
            stage_frozen_copy(source, stage / relative, digest)
        # Recheck both model families and original inventory before canonical mutation.
        assert_compiled_unchanged(assets + naf_assets)
        assert_file_unchanged(inventory_path, old_inventory_digest)
        assert_file_unchanged(naf_notice, naf_notice_digest)
        for path, digest in package_files:
            assert_file_unchanged(path, digest)
        verify_official_sources()
        for asset, _ in assets:
            if not (models / asset.name).exists():
                os.replace(stage / asset.name, models / asset.name)
        for _, relative, _ in copies:
            (models / relative).parent.mkdir(parents=True, exist_ok=True)
            if not (models / relative).exists():
                os.replace(stage / relative, models / relative)
        # Existing destinations are also checked here: a concurrent change must never
        # be endorsed by publishing a new inventory after the preflight comparison.
        assert_compiled_unchanged(assets + naf_assets + [(models / asset.name, identity) for asset, identity in assets])
        for source, relative, digest in copies:
            assert_file_unchanged(source, digest)
            assert_file_unchanged(models / relative, digest)
        assert_file_unchanged(inventory_path, old_inventory_digest)
        assert_file_unchanged(naf_notice, naf_notice_digest)
        for path, digest in package_files:
            assert_file_unchanged(path, digest)
        temporary_inventory = models / 'Provenance/.model-inventory-cgnet.json'
        temporary_inventory.write_text(json.dumps(inventory, indent=2) + '\n')
        os.replace(temporary_inventory, models / 'Provenance/model-inventory.json')
    print('PASS: promoted four validated assets and three frozen modes; NAFNet pair preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled-root', type=Path, required=True)
    parser.add_argument('--converted-root', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'Targets/TruongNoiserApp/Models')
    promote(parser.parse_args())


if __name__ == '__main__':
    main()
