"""Negative bundle-layout checks, with no model execution or production substitutions.

The real app/resource audits still run on compiled products. These tiny file-layout fixtures
exercise refusal paths without needing to mutate an installed or signed application.
"""
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BuiltMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TruongNoiser-audit-test-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.app = root / "Fixture.app"
        self.app.mkdir()
        info = dict(CFBundleIdentifier="com.iantruong.truongnoiser", CFBundleDisplayName="TruongNoiser",
            CFBundleExecutable="TruongNoiserApp", MinimumOSVersion="27.0", NSAccentColorName="AccentColor",
            NSSupportsLiveActivities=True,
            BGTaskSchedulerPermittedIdentifiers=["com.iantruong.truongnoiser.continued-processing.*"],
            CFBundleURLTypes=[dict(CFBundleURLSchemes=["truongnoiser"])], LSApplicationQueriesSchemes=["truongscaler"],
            UISupportedInterfaceOrientations=["UIInterfaceOrientationPortrait", "UIInterfaceOrientationLandscapeLeft", "UIInterfaceOrientationLandscapeRight"])
        info.update(
            LSSupportsOpeningDocumentsInPlace=False,
            CFBundleDocumentTypes=[dict(CFBundleTypeRole="Viewer", LSHandlerRank="Owner",
                LSItemContentTypes=["com.iantruong.imagingsuite.handoff.noiser"])],
            UTExportedTypeDeclarations=[dict(UTTypeIdentifier="com.iantruong.imagingsuite.handoff.noiser",
                UTTypeTagSpecification={"public.filename-extension": ["truongnoiserhandoff"]})],
            UTImportedTypeDeclarations=[dict(UTTypeIdentifier="com.iantruong.imagingsuite.handoff.scaler",
                UTTypeTagSpecification={"public.filename-extension": ["truongscalerhandoff"]})])
        (self.app / "Info.plist").write_bytes(plistlib.dumps(info))
        (self.app / "TruongNoiserApp").write_bytes(b"layout-fixture-executable")
        (self.app / "Assets.car").write_bytes(b"layout-fixture-catalog")
        widget = self.app / "PlugIns/TruongNoiserLiveActivityExtension.appex"
        widget.mkdir(parents=True)
        (widget / "Info.plist").write_bytes(plistlib.dumps(dict(CFBundleIdentifier="com.iantruong.truongnoiser.liveactivity",
            NSExtension=dict(NSExtensionPointIdentifier="com.apple.widgetkit-extension"))))
        (widget / "TruongNoiserLiveActivityExtension").write_bytes(b"layout-fixture-widget")
        models = self.app / "Models"
        for family in ["NAFNet_w64", "CGNet_SIDD"]:
            for size in [256, 512]:
                asset = models / f"{family}_{size}.mlmodelc"
                for name in ["coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin", "analytics/coremldata.bin"]:
                    target = asset / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"layout-fixture-only")
        for notice in ["NAFNet-LICENSE.txt", "CGNet-LICENSE.txt"]:
            target = models / "Notices" / notice
            target.parent.mkdir(exist_ok=True)
            target.write_text("layout-fixture-notice")
        provenance = models / "Provenance"
        provenance.mkdir()
        (provenance / "checkpoint-sha256.txt").write_text("layout-fixture-checksums")
        identifiers = [f"{family}_{size}" for family in ["NAFNet_w64", "CGNet_SIDD"] for size in [256, 512]]
        (provenance / "model-inventory.json").write_text(json.dumps(dict(canonicalModelCount=4,
            models=[dict(identifier=name, asset=name + ".mlmodelc") for name in identifiers])))
        self.assetutil = root / "assetutil-fixture"
        self.assetutil.write_text('#!/bin/sh\nprintf \'"Name" : "AccentColor"\\n"System Color Name" : "systemGreenColor"\\n\'\n')
        self.assetutil.chmod(0o700)

    def audit(self):
        return subprocess.run([sys.executable, str(ROOT / "scripts/validate_built_metadata.py"), "--app", str(self.app),
            "--assetutil", str(self.assetutil)], capture_output=True, text=True)

    def testCompleteLayoutPasses(self):
        result = self.audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def testMissingPortableDocumentDeclarationsAreRefused(self):
        path = self.app / "Info.plist"
        original = plistlib.loads(path.read_bytes())
        for key in ("CFBundleDocumentTypes", "UTExportedTypeDeclarations", "UTImportedTypeDeclarations"):
            with self.subTest(key=key):
                info = dict(original)
                del info[key]
                path.write_bytes(plistlib.dumps(info))
                self.assertNotEqual(self.audit().returncode, 0)
        path.write_bytes(plistlib.dumps(original))

    def testPortableDocumentOpeningInPlaceIsRefused(self):
        path = self.app / "Info.plist"
        info = plistlib.loads(path.read_bytes())
        info["LSSupportsOpeningDocumentsInPlace"] = True
        path.write_bytes(plistlib.dumps(info))
        self.assertNotEqual(self.audit().returncode, 0)

    def testMissingMainExecutableIsRefused(self):
        (self.app / "TruongNoiserApp").unlink()
        self.assertNotEqual(self.audit().returncode, 0)

    def testEmptyMainExecutableIsRefused(self):
        (self.app / "TruongNoiserApp").write_bytes(b"")
        self.assertNotEqual(self.audit().returncode, 0)

    def testEmbeddedHostedTestBundleIsRefused(self):
        (self.app / "PlugIns/TruongNoiserHostTests.xctest").mkdir()
        self.assertNotEqual(self.audit().returncode, 0)

    def testMissingCGNetShapeIsRefused(self):
        (self.app / "Models/CGNet_SIDD_512.mlmodelc/weights/weight.bin").unlink()
        self.assertNotEqual(self.audit().returncode, 0)

    def testRetiredCompiledModelIsRefused(self):
        (self.app / "Models/Retired_x4_256.mlmodelc").mkdir()
        self.assertNotEqual(self.audit().returncode, 0)

    def testTwoModelInventoryCannotDescribeAFourAssetBundle(self):
        path = self.app / "Models/Provenance/model-inventory.json"
        inventory = json.loads(path.read_text())
        inventory["canonicalModelCount"] = 2
        path.write_text(json.dumps(inventory))
        self.assertNotEqual(self.audit().returncode, 0)

    def testEachTrainedFamilyNoticeIsRequired(self):
        for name in ["NAFNet-LICENSE.txt", "CGNet-LICENSE.txt"]:
            path = self.app / "Models/Notices" / name
            original = path.read_bytes()
            path.unlink()
            self.assertNotEqual(self.audit().returncode, 0, name)
            path.write_bytes(original)



if __name__ == "__main__":
    unittest.main()
