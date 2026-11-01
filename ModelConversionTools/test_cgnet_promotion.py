"""Promotion must reject plausible averages backed by incomplete/failed raw evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from bundle_cgnet import (
    assert_compiled_unchanged, load_json_snapshot, stage_frozen_copy, validate_raw_summary, validate_report,
)
from cgnet_reference import CHECKPOINT_SHA256, COMMIT
from verify_cgnet_parity import CASE_NAMES, compiled_identity


def report_fixture():
    rows = [{'quarterTurnsCCW': turn, 'mirrorHorizontalBeforeRotation': mirror,
        'reference': {'finite': True, 'requiresProductionRefusal': False, 'minFinite': .1, 'maxFinite': .9},
        'prediction': {'finite': True, 'requiresProductionRefusal': False, 'minFinite': .1, 'maxFinite': .9},
        'psnrDB': 75., 'maxAbsoluteError': .0005}
        for mirror in (False, True) for turn in range(4)]
    return {'model': 'CGNet-SIDD', 'size': 256, 'computeRequested': 'cpuOnly', 'ensemble': 'x8',
        'officialCommit': COMMIT, 'checkpointSHA256': CHECKPOINT_SHA256,
        'inputShape': [1, 3, 256, 256], 'outputShape': [1, 3, 256, 256], 'tensorType': 'FLOAT16',
        'passed': True, 'failures': [], 'compiledAssetUnchanged': True,
        'compiledDirectorySHA256': 'a' * 64, 'compiledFiles': [],
        'strengthsApplied': False, 'outputClamped': False,
        'fidelityGate': {'psnrMinDB': 50, 'maxAbsoluteErrorExclusive': .02},
        'productionRawRange': [-.25, 1.25],
        'cases': [{'case': name, 'orientations': copy.deepcopy(rows), 'status': 'PASS',
            'psnrDB': 80., 'maxAbsoluteError': .0003} for name in CASE_NAMES],
        'finiteFidelityComparisons': len(CASE_NAMES), 'refusalObservations': 0,
        'warmPredictionSeconds': [.3]}


class CGNetPromotionTests(unittest.TestCase):
    def validate(self, report):
        validate_report(report, 256, 'cpuOnly', 'x8', ('a' * 64, []))

    def testCompleteRawAndFinalEvidencePasses(self):
        self.validate(report_fixture())

    def testPassingAverageCannotHideFailingRawOrientation(self):
        for key, value in [('psnrDB', 49.99), ('maxAbsoluteError', .02), ('maxAbsoluteError', float('nan'))]:
            report = report_fixture(); report['cases'][0]['orientations'][7][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.validate(report)

    def testPartialOrRepeatedOrientationsCannotPass(self):
        for replacement in (report_fixture()['cases'][0]['orientations'][:7],
                            [report_fixture()['cases'][0]['orientations'][0]] * 8):
            report = report_fixture(); report['cases'][0]['orientations'] = replacement
            with self.assertRaises(RuntimeError): self.validate(report)

    def testWrongAssetPolicyModeOrChangedBytesCannotPass(self):
        for key, value in [('compiledDirectorySHA256', 'b' * 64), ('compiledAssetUnchanged', False),
                           ('computeRequested', 'all'), ('ensemble', 'none'), ('checkpointSHA256', 'b' * 64)]:
            report = report_fixture(); report[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.validate(report)

    def testMissingHighFrequencyCaseCannotPass(self):
        report = report_fixture(); report['cases'].pop()
        with self.assertRaises(RuntimeError): self.validate(report)

    def testRefusalIsRecordedButNotCountedAsFidelity(self):
        report = report_fixture(); case = report['cases'][0]
        row = case['orientations'][0]
        row['reference']['requiresProductionRefusal'] = True
        row['prediction']['requiresProductionRefusal'] = True
        row['reference']['minFinite'] = -1
        row['prediction']['minFinite'] = -1
        case['status'] = 'REQUIRES_SAFE_REFUSAL'; case['rawConvertedOutputRequiresRefusal'] = True
        report['finiteFidelityComparisons'] -= 1; report['refusalObservations'] += 1
        self.validate(report)
        row['prediction']['requiresProductionRefusal'] = False
        with self.assertRaises(RuntimeError): self.validate(report)

    def testIdentityClampOrNoActualWarmPredictionCannotPass(self):
        for key, value in [('outputClamped', True), ('strengthsApplied', True), ('warmPredictionSeconds', [])]:
            report = report_fixture(); report[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.validate(report)


    def testExplicitFiniteAndRefusalFlagsCannotBeOmittedOrReplacedByNumbers(self):
        for side in ('reference', 'prediction'):
            for key in ('finite', 'requiresProductionRefusal'):
                for replacement in (None, 0, 1):
                    report = report_fixture()
                    state = report['cases'][0]['orientations'][0][side]
                    if replacement is None: del state[key]
                    else: state[key] = replacement
                    with self.subTest(side=side, key=key, value=replacement), self.assertRaises(RuntimeError):
                        self.validate(report)

    def testClaimedSafeRawExtremaMustBeFiniteOrderedAndWithinTheProductionRange(self):
        for side in ('reference', 'prediction'):
            for key, value in [('minFinite', -100), ('maxFinite', 100), ('minFinite', 1),
                               ('maxFinite', float('nan')), ('minFinite', None), ('maxFinite', True)]:
                report = report_fixture(); report['cases'][0]['orientations'][0][side][key] = value
                with self.subTest(side=side, key=key, value=value), self.assertRaises(RuntimeError):
                    self.validate(report)
        for limits in (None, [0, 1], [-100, 100]):
            report = report_fixture()
            if limits is None: del report['productionRawRange']
            else: report['productionRawRange'] = limits
            with self.subTest(limits=limits), self.assertRaises(RuntimeError): self.validate(report)

    def testNonfiniteRawStateCanOnlyRecordAnExplicitRefusal(self):
        for extrema in ((None, None), (.2, .8)):
            state = {'finite': False, 'requiresProductionRefusal': True,
                     'minFinite': extrema[0], 'maxFinite': extrema[1]}
            validate_raw_summary(state)
            state['requiresProductionRefusal'] = False
            with self.assertRaises(RuntimeError): validate_raw_summary(state)
        with self.assertRaises(RuntimeError):
            validate_raw_summary({'finite': False, 'requiresProductionRefusal': True})


class FrozenPromotionFilesTests(unittest.TestCase):
    def testSourceMutationAfterPreflightCannotBecomeNewlyAcceptedEvidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / 'report.json', Path(temporary) / 'stage/report.json'
            source.write_text(json.dumps({'passed': True}))
            report, digest = load_json_snapshot(source)
            self.assertTrue(report['passed'])
            source.write_text(json.dumps({'passed': False}))
            with self.assertRaisesRegex(RuntimeError, 'Preflight file changed'):
                stage_frozen_copy(source, destination, digest)
            self.assertFalse(destination.exists())

    def testSourceOrCopyMutationDuringStagingIsBoundToThePreflightDigest(self):
        copyfile = shutil.copyfile
        for change_source in (False, True):
            with self.subTest(change_source=change_source), tempfile.TemporaryDirectory() as temporary:
                source, destination = Path(temporary) / 'report.json', Path(temporary) / 'stage/report.json'
                source.write_text(json.dumps({'passed': True}))
                _, digest = load_json_snapshot(source)
                def changing_copy(source_path, destination_path):
                    copyfile(source_path, destination_path)
                    (source_path if change_source else destination_path).write_text(json.dumps({'passed': False}))
                with patch('bundle_cgnet.shutil.copyfile', side_effect=changing_copy):
                    with self.assertRaisesRegex(RuntimeError, 'Preflight file changed'):
                        stage_frozen_copy(source, destination, digest)

    def testUnchangedStagedBytesAndBothModelFamiliesRetainTheirIdentity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'report.json', root / 'stage/report.json'
            source.write_text(json.dumps({'passed': True})); _, digest = load_json_snapshot(source)
            stage_frozen_copy(source, destination, digest)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            assets = []
            for family in ('NAFNet_w64', 'CGNet_SIDD'):
                asset = root / (family + '.mlmodelc'); asset.mkdir()
                (asset / 'weight.bin').write_bytes(family.encode())
                assets.append((asset, compiled_identity(asset)))
            assert_compiled_unchanged(assets)
            for asset, _ in assets:
                weight = asset / 'weight.bin'; old = weight.read_bytes(); weight.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Compiled model changed'):
                    assert_compiled_unchanged(assets)
                weight.write_bytes(old)


if __name__ == '__main__':
    unittest.main()
