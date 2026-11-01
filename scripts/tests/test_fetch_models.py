"""Small filesystem/archive safety regressions; no weights, network or conversion."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "fetch-models.py"
SPEC = importlib.util.spec_from_file_location("fetch_models", SCRIPT)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)
MODEL = "Targets/TruongNoiserApp/Models/Fixture.mlmodelc/weights/weight.bin"
METADATA = "Targets/TruongNoiserApp/Models/Fixture.mlmodelc/metadata.json"
CHECKPOINT = "ModelConversionTools/checkpoints/Fixture.pth"


class FetchModelsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="Noiser-fetch-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkout"
        self.root.mkdir()
        self.files = {"models": {MODEL: b"small-trained-fixture", METADATA: b'{"shape":[1,3,256,256]}'},
                      "checkpoints": {CHECKPOINT: b"official-checkpoint-fixture"}}
        self.payloads = {}
        self.manifest = {"schemaVersion": 1, "repository": "iant4093/TruongNoiser",
                         "releaseTag": "model-assets-v1", "archives": []}
        for kind in ("models", "checkpoints"):
            name = f"Noiser-{kind}-v1.zip"
            self.manifest["archives"].append({"kind": kind, "name": name,
                "url": f"https://github.com/iant4093/TruongNoiser/releases/download/model-assets-v1/{name}",
                "files": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for path, data in self.files[kind].items()]})
            self.set_archive(kind, list(self.files[kind].items()))
        self.downloads = []

    def set_archive(self, kind, entries):
        data = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in entries:
                    archive.writestr(name, content)
        payload = data.getvalue()
        self.payloads[kind] = payload
        row = next(a for a in self.manifest["archives"] if a["kind"] == kind)
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def download(self, archive, destination):
        self.downloads.append(archive["kind"])
        destination.write_bytes(self.payloads[archive["kind"]])

    def hydrate(self, **kwargs):
        return FETCH.hydrate(self.root, self.manifest, download=self.download, **kwargs)

    def install(self, kind="models"):
        for relative, data in self.files[kind].items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def assert_no_model_published(self):
        self.assertFalse((self.root / MODEL).exists())
        self.assertFalse((self.root / METADATA).exists())

    def test_missing_models_hydrate_byte_exactly_and_checkpoints_are_optional(self):
        self.assertEqual(self.hydrate(), 2)
        self.assertEqual(self.downloads, ["models"])
        for path, data in self.files["models"].items():
            self.assertEqual((self.root / path).read_bytes(), data)
        self.assertFalse((self.root / CHECKPOINT).exists())
        self.assertEqual(self.hydrate(checkpoints=True), 3)
        self.assertEqual(self.downloads, ["models", "checkpoints"])
        self.assertEqual((self.root / CHECKPOINT).read_bytes(), self.files["checkpoints"][CHECKPOINT])

    def test_complete_valid_assets_work_offline_without_touching_files(self):
        self.install()
        before = {p: (self.root / p).stat() for p in self.files["models"]}
        self.hydrate()
        self.assertEqual(self.downloads, [])
        self.assertFalse((self.root / "build").exists())
        for path, info in before.items():
            self.assertEqual((self.root / path).stat().st_ino, info.st_ino)
            self.assertEqual((self.root / path).stat().st_mtime_ns, info.st_mtime_ns)

    def test_check_missing_never_downloads_or_creates_directories(self):
        with self.assertRaises(FETCH.AssetError):
            self.hydrate(check=True)
        self.assertEqual(self.downloads, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_check_complete_is_read_only(self):
        self.install()
        self.assertEqual(self.hydrate(check=True), 2)
        self.assertEqual(self.downloads, [])
        self.assertFalse((self.root / "build").exists())

    def test_changed_existing_asset_is_preserved_before_any_download(self):
        changed = self.root / METADATA
        changed.parent.mkdir(parents=True)
        changed.write_bytes(b"human edit")
        with self.assertRaisesRegex(FETCH.AssetError, "preserved"):
            self.hydrate()
        self.assertEqual(changed.read_bytes(), b"human edit")
        self.assertEqual(self.downloads, [])
        self.assertFalse((self.root / MODEL).exists())

    def test_resume_only_missing_files_from_verified_cache_preserves_valid_files(self):
        self.hydrate()
        valid = (self.root / METADATA).stat()
        (self.root / MODEL).unlink()
        self.hydrate()
        self.assertEqual(self.downloads, ["models"])
        self.assertEqual((self.root / MODEL).read_bytes(), self.files["models"][MODEL])
        self.assertEqual((self.root / METADATA).stat().st_ino, valid.st_ino)
        self.assertEqual((self.root / METADATA).stat().st_mtime_ns, valid.st_mtime_ns)

    def test_wrong_download_archive_hash_never_publishes(self):
        self.payloads["models"] += b"unexpected bytes"
        with self.assertRaisesRegex(FETCH.AssetError, "size/SHA-256"):
            self.hydrate()
        self.assert_no_model_published()

    def test_wrong_member_hash_never_publishes(self):
        self.manifest["archives"][0]["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(FETCH.AssetError, "member SHA-256"):
            self.hydrate()
        self.assert_no_model_published()

    def test_missing_declared_member_never_publishes(self):
        self.set_archive("models", [(MODEL, self.files["models"][MODEL])])
        with self.assertRaisesRegex(FETCH.AssetError, "missing declared"):
            self.hydrate()
        self.assert_no_model_published()

    def test_undeclared_member_never_publishes(self):
        self.set_archive("models", list(self.files["models"].items()) + [("unrelated.txt", b"extra")])
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assert_no_model_published()
        self.assertFalse((self.root / "unrelated.txt").exists())

    def test_oversized_member_never_publishes(self):
        self.set_archive("models", [(MODEL, self.files["models"][MODEL] + b"extra"),
                                    (METADATA, self.files["models"][METADATA])])
        with self.assertRaisesRegex(FETCH.AssetError, "incorrectly sized"):
            self.hydrate()
        self.assert_no_model_published()

    def test_duplicate_member_never_publishes(self):
        self.set_archive("models", list(self.files["models"].items()) + [(MODEL, self.files["models"][MODEL])])
        with self.assertRaisesRegex(FETCH.AssetError, "Duplicate"):
            self.hydrate()
        self.assert_no_model_published()

    def test_zip_path_escape_is_refused(self):
        self.set_archive("models", [("../outside.txt", b"escaped")])
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assertFalse((self.root.parent / "outside.txt").exists())
        self.assert_no_model_published()

    def test_zip_absolute_path_is_refused(self):
        outside = self.root.parent / "outside.txt"
        self.set_archive("models", [(str(outside), b"escaped")])
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assertFalse(outside.exists())

    def test_zip_symlink_is_refused(self):
        link = zipfile.ZipInfo(MODEL)
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.set_archive("models", [(link, b"../../outside")])
        with self.assertRaisesRegex(FETCH.AssetError, "non-regular"):
            self.hydrate()
        self.assert_no_model_published()

    def test_symlink_parent_is_refused_without_touching_its_destination(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        (self.root / "Targets").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.downloads, [])

    def test_symlink_asset_is_refused_without_following_it(self):
        outside = self.root.parent / "outside.bin"
        outside.write_bytes(self.files["models"][MODEL])
        target = self.root / MODEL
        target.parent.mkdir(parents=True)
        target.symlink_to(outside)
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assertTrue(target.is_symlink())
        self.assertEqual(outside.read_bytes(), self.files["models"][MODEL])
        self.assertEqual(self.downloads, [])

    def test_symlink_cache_is_refused(self):
        outside = self.root.parent / "outside.zip"
        outside.write_bytes(self.payloads["models"])
        cache = self.root / "build/model-assets/Noiser-models-v1.zip"
        cache.parent.mkdir(parents=True)
        cache.symlink_to(outside)
        with self.assertRaises(FETCH.AssetError):
            self.hydrate()
        self.assertTrue(cache.is_symlink())
        self.assertEqual(self.downloads, [])
        self.assert_no_model_published()

    def test_changed_during_download_refuses_before_any_publication(self):
        def download(archive, destination):
            self.download(archive, destination)
            path = self.root / METADATA
            path.parent.mkdir(parents=True)
            path.write_bytes(b"new local edit")
        with self.assertRaisesRegex(FETCH.AssetError, "preserved"):
            FETCH.hydrate(self.root, self.manifest, download=download)
        self.assertFalse((self.root / MODEL).exists())
        self.assertEqual((self.root / METADATA).read_bytes(), b"new local edit")

    def test_invalid_manifest_scope_and_oversize_are_refused_before_download(self):
        for path in ("../human.txt", "/human.txt", "Targets/elsewhere/data.bin", "Targets//TruongNoiserApp/Models/file.bin"):
            manifest = copy.deepcopy(self.manifest)
            manifest["archives"][0]["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(FETCH.AssetError):
                FETCH.hydrate(self.root, manifest, download=self.download)
        manifest = copy.deepcopy(self.manifest)
        manifest["archives"][0]["files"][0]["bytes"] = FETCH.MAX_BYTES + 1
        with self.assertRaises(FETCH.AssetError):
            FETCH.hydrate(self.root, manifest, download=self.download)
        self.assertEqual(self.downloads, [])


if __name__ == "__main__":
    unittest.main()
