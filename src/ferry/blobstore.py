"""Pluggable blob storage for checkpoints and persisted node state (DESIGN §6)."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urlparse


class BlobStore(Protocol):
    def put(self, key: str, data: bytes | BinaryIO) -> str:
        """Store ``data`` under ``key`` atomically and return its URI."""
        ...

    def get(self, uri: str) -> BinaryIO: ...

    def delete(self, uri: str) -> None: ...


def _read_all(data: bytes | BinaryIO) -> bytes:
    return data if isinstance(data, bytes | bytearray | memoryview) else data.read()


class LocalFSBlobStore:
    """Filesystem store. ``put`` writes to a temp file and renames, so readers never see a partial blob."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"key escapes blob root: {key!r}")
        return p

    def _from_uri(self, uri: str, *, confined: bool = True) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in ("file", ""):
            raise ValueError(f"not a file:// URI: {uri}")
        p = Path(parsed.path).resolve()
        if confined and self.root not in p.parents:
            raise ValueError(f"URI outside blob root: {uri}")
        return p

    def put(self, key: str, data: bytes | BinaryIO) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_read_all(data))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path.as_uri()

    def get(self, uri: str) -> BinaryIO:
        # Reads may point at any file:// URI handed out by the coordinator (e.g. a shared mount at another path).
        return open(self._from_uri(uri, confined=False), "rb")

    def delete(self, uri: str) -> None:
        self._from_uri(uri).unlink(missing_ok=True)


class S3BlobStore:
    """S3 store (``pip install ferry[s3]``). S3 PUTs are atomic per key."""

    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        if client is None:
            import boto3  # optional dependency

            client = boto3.client("s3")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes | BinaryIO) -> str:
        k = self._key(key)
        self.client.put_object(Bucket=self.bucket, Key=k, Body=_read_all(data))
        return f"s3://{self.bucket}/{k}"

    def get(self, uri: str) -> BinaryIO:
        p = urlparse(uri)
        obj = self.client.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
        return io.BytesIO(obj["Body"].read())

    def delete(self, uri: str) -> None:
        p = urlparse(uri)
        self.client.delete_object(Bucket=p.netloc, Key=p.path.lstrip("/"))


def open_blob_store(uri: str) -> BlobStore:
    """Build a store from a config URI: ``file://./ferry-data`` or ``s3://bucket/prefix``."""
    p = urlparse(uri)
    if p.scheme == "file":
        return LocalFSBlobStore((p.netloc + p.path) or ".")
    if p.scheme == "s3":
        return S3BlobStore(p.netloc, p.path)
    if p.scheme == "":
        return LocalFSBlobStore(uri)
    raise ValueError(f"unsupported blob store URI: {uri}")


def read_blob(store: BlobStore, uri: str) -> bytes:
    with store.get(uri) as f:
        return f.read()
