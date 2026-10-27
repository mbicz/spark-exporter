#!/usr/bin/python3
"""Prepare pinned official binaries; Debian builds always invoke offline mode."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request


VERSION = "1.12.1"
BASE_URL = "https://github.com/prometheus/node_exporter/releases/download/v" + VERSION
ROOT = Path(__file__).resolve().parent.parent


def checksums(text):
    return {name: digest for digest, name in (line.split() for line in text.splitlines() if line.strip())}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(name, destination):
    with urllib.request.urlopen(BASE_URL + "/" + name, timeout=60) as response:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            try:
                shutil.copyfileobj(response, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    os.replace(str(temporary), str(destination))


def prepare(args):
    release = "node_exporter-{}.linux-{}".format(VERSION, args.architecture)
    archive_name = release + ".tar.gz"
    pinned = checksums((ROOT / "debian/node-exporter.sha256").read_text(encoding="ascii"))[archive_name]
    cache = ROOT / "vendor/cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive).resolve() if args.archive else cache / archive_name
    if not args.offline and not args.archive:
        manifest = cache / "sha256sums.txt"
        download("sha256sums.txt", manifest)
        published = checksums(manifest.read_text(encoding="ascii"))
        if published.get(archive_name) != pinned:
            raise ValueError("upstream checksum differs from the reviewed pin for " + archive_name)
        download(archive_name, archive)
    if not archive.is_file():
        raise ValueError("missing pinned archive; run make vendor before the offline Debian build")
    if digest(archive) != pinned:
        raise ValueError("SHA-256 mismatch for " + str(archive))
    # Retain a verified archive for a subsequent fully offline dpkg-buildpackage.
    cached_archive = cache / archive_name
    if archive != cached_archive:
        shutil.copyfile(str(archive), str(cached_archive))
    destination = ROOT / "vendor" / ("linux-" + args.architecture)
    destination.mkdir(parents=True, exist_ok=True)
    files = {"node_exporter": "node_exporter", "LICENSE": "LICENSE.node_exporter", "NOTICE": "NOTICE.node_exporter"}
    with tempfile.TemporaryDirectory(dir=str(destination.parent)) as temporary:
        staging = Path(temporary)
        with tarfile.open(str(cached_archive), "r:gz") as bundle:
            for source_name, installed_name in files.items():
                member = bundle.getmember(release + "/" + source_name)
                if not member.isfile():
                    raise ValueError("release member is not a regular file: " + member.name)
                with bundle.extractfile(member) as source, (staging / installed_name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                (staging / installed_name).chmod(0o755 if source_name == "node_exporter" else 0o644)
        for name in files.values():
            os.replace(str(staging / name), str(destination / name))
    print("Prepared verified node_exporter {} for linux-{}".format(VERSION, args.architecture))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("amd64", "arm64"))
    parser.add_argument("--archive", help="use a local official release archive, without network access")
    parser.add_argument("--offline", action="store_true", help="require an already cached or supplied archive")
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, KeyError, tarfile.TarError, urllib.error.URLError) as error:
        parser.exit(1, "node_exporter preparation failed: {}\n".format(error))


if __name__ == "__main__":
    main()
