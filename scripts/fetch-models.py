#!/usr/bin/env python3
"""Hydrate the exact validated model bytes from a pinned public release asset.

Complete valid local assets work offline. Existing changed files are refused, never
replaced. An archive is fully verified in staging before any missing file is linked
atomically into the checkout. No model conversion or production fallback is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import urllib.request
import zipfile

CHUNK = 1024 * 1024
MAX_BYTES = 8 * 1024 ** 3
PREFIXES = {
    "models": "Targets/TruongNoiserApp/Models/",
    "checkpoints": "ModelConversionTools/checkpoints/",
}


class AssetError(RuntimeError):
    pass


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise AssetError("Invalid asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise AssetError("Asset paths must be canonical relative paths")
    return path


def _digest(stream):
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        count += len(block)
        digest.update(block)
    return count, digest.hexdigest()


def validate_manifest(manifest):
    if manifest.get("schemaVersion") != 1:
        raise AssetError("Unsupported model-assets manifest")
    if manifest.get("repository") != "iant4093/TruongNoiser" or manifest.get("releaseTag") != "model-assets-v1":
        raise AssetError("Unexpected model asset release")
    archives = manifest.get("archives", [])
    if len(archives) != 2 or {a.get("kind") for a in archives} != set(PREFIXES):
        raise AssetError("Expected one models archive and one checkpoints archive")
    all_paths = set()
    for archive in archives:
        kind = archive["kind"]
        name = archive.get("name", "")
        if _relative(name).name != name or not name.endswith(".zip"):
            raise AssetError("Invalid archive filename")
        expected_url = f"https://github.com/iant4093/TruongNoiser/releases/download/model-assets-v1/{name}"
        if archive.get("url") != expected_url:
            raise AssetError("Unexpected archive URL")
        if type(archive.get("bytes")) is not int or not 0 < archive["bytes"] <= MAX_BYTES:
            raise AssetError("Missing or invalid pinned archive size")
        if not re.fullmatch(r"[0-9a-f]{64}", archive.get("sha256", "")):
            raise AssetError("Missing or invalid pinned archive SHA-256")
        files = archive.get("files", [])
        if not files or len(files) > 10000:
            raise AssetError("Invalid archive inventory")
        total = 0
        for entry in files:
            name = entry.get("path")
            _relative(name)
            if not name.startswith(PREFIXES[kind]) or name in all_paths:
                raise AssetError("Duplicate or out-of-scope asset path")
            all_paths.add(name)
            if type(entry.get("bytes")) is not int or not 0 <= entry["bytes"] <= MAX_BYTES:
                raise AssetError("Invalid declared asset size")
            if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
                raise AssetError("Invalid declared asset SHA-256")
            total += entry["bytes"]
        if total > MAX_BYTES:
            raise AssetError("Archive uncompressed inventory exceeds the safety bound")
    return archives


@contextmanager
def _parent_fd(root, relative, create=False):
    """Walk anchored directory descriptors without following a symbolic link."""
    parts = _relative(relative).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            if create:
                try:
                    os.mkdir(component, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def _state(root, entry):
    try:
        with _parent_fd(root, entry["path"]) as (parent, name):
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise AssetError("Refusing a non-regular asset: " + entry["path"])
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                actual = _digest(stream)
    except FileNotFoundError:
        return "missing"
    except OSError as error:
        raise AssetError("Unsafe or unreadable asset path: " + entry["path"]) from error
    if actual != (entry["bytes"], entry["sha256"]):
        raise AssetError("Existing asset differs; preserved without changes: " + entry["path"])
    return "valid"


def _matches_archive(path, archive):
    if path.is_symlink():
        raise AssetError("Refusing a symbolic link in the model cache")
    if not path.exists():
        return False
    if not path.is_file():
        raise AssetError("Refusing a non-file model cache entry")
    with path.open("rb") as stream:
        return _digest(stream) == (archive["bytes"], archive["sha256"])


def _download(archive, destination):
    request = urllib.request.Request(archive["url"], headers={"User-Agent": "TruongNoiser-model-fetch/1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        count = 0
        for block in iter(lambda: response.read(CHUNK), b""):
            count += len(block)
            if count > archive["bytes"]:
                raise AssetError("Download exceeds its pinned size")
            output.write(block)
        output.flush()
        os.fsync(output.fileno())


def _archive(root, archive, download):
    relative = "build/model-assets/" + archive["name"]
    with _parent_fd(root, relative, create=True):
        pass
    destination = root / relative
    if _matches_archive(destination, archive):
        return destination
    descriptor, temporary = tempfile.mkstemp(prefix=archive["name"] + ".", suffix=".partial",
                                              dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        download(archive, temporary)
        if not _matches_archive(temporary, archive):
            raise AssetError("Downloaded archive size/SHA-256 does not match its pin")
        if destination.is_symlink():
            raise AssetError("Refusing a symbolic link in the model cache")
        os.replace(temporary, destination)
        with _parent_fd(root, relative) as (parent, _):
            os.fsync(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _stage(archive_path, archive, staging):
    expected = {entry["path"]: entry for entry in archive["files"]}
    seen = set()
    with zipfile.ZipFile(archive_path) as source:
        for member in source.infolist():
            directory = member.is_dir()
            name = member.filename[:-1] if directory else member.filename
            _relative(name)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if member.filename in seen or member.flag_bits & 1 or mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise AssetError("Duplicate, encrypted or non-regular archive member")
            seen.add(member.filename)
            if directory:
                if mode == stat.S_IFREG or not any(p.startswith(name + "/") for p in expected):
                    raise AssetError("Undeclared archive directory")
                continue
            if mode == stat.S_IFDIR or name not in expected or member.file_size != expected[name]["bytes"]:
                raise AssetError("Undeclared or incorrectly sized archive member: " + name)
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with source.open(member) as content, destination.open("xb") as output:
                for block in iter(lambda: content.read(CHUNK), b""):
                    count += len(block)
                    if count > expected[name]["bytes"]:
                        raise AssetError("Archive member exceeds its pinned size")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if (count, digest.hexdigest()) != (expected[name]["bytes"], expected[name]["sha256"]):
                raise AssetError("Archive member SHA-256 mismatch: " + name)
    if {name for name in seen if not name.endswith("/")} != set(expected):
        raise AssetError("Archive is missing declared assets")


def hydrate(root, manifest, *, checkpoints=False, check=False, download=_download):
    root = Path(root).resolve(strict=True)
    selected = [a for a in validate_manifest(manifest) if a["kind"] == "models" or checkpoints]
    states = {entry["path"]: _state(root, entry) for a in selected for entry in a["files"]}
    missing = [name for name, state in states.items() if state == "missing"]
    if check and missing:
        raise AssetError("Missing pinned assets: " + ", ".join(missing))
    for archive in selected:
        if all(states[e["path"]] == "valid" for e in archive["files"]):
            continue
        archive_path = _archive(root, archive, download)
        with tempfile.TemporaryDirectory(prefix=".verified-", dir=archive_path.parent) as directory:
            staging = Path(directory)
            _stage(archive_path, archive, staging)
            # Recheck every existing file before publishing any missing file.
            for entry in archive["files"]:
                _state(root, entry)
            for entry in archive["files"]:
                if _state(root, entry) == "valid":
                    continue
                with _parent_fd(root, entry["path"], create=True) as (parent, name):
                    try:
                        os.link(staging / entry["path"], name, dst_dir_fd=parent, follow_symlinks=False)
                        os.fsync(parent)
                    except FileExistsError:
                        _state(root, entry)
    for archive in selected:
        for entry in archive["files"]:
            if _state(root, entry) != "valid":
                raise AssetError("Asset publication incomplete: " + entry["path"])
    return len(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", action="store_true", help="Also fetch/verify the official reference checkpoints")
    parser.add_argument("--check", action="store_true", help="Verify local files only; never download or create files")
    args = parser.parse_args()
    try:
        manifest = json.loads(Path(__file__).with_name("model-assets.json").read_text())
        count = hydrate(Path(__file__).resolve().parents[1], manifest,
                        checkpoints=args.checkpoints, check=args.check)
    except (AssetError, OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"PASS: {count} pinned model/reference files verified; existing valid assets work offline")


if __name__ == "__main__":
    main()
