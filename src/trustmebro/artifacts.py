"""Length-prefixed binary records; callers own compression and publication.

The archive families use different byte orders, so the header layout is explicit.
"""

import hashlib
import sys
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from typing import Literal, Protocol

import msgspec

from trustmebro.extraction.records import SchemaError

_BIG_NAT_EXT = 1
_MAX_MSGPACK_NAT = 2**64 - 1
_encoder = msgspec.msgpack.Encoder()


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> object: ...


def write_frame(stream: BinaryWriter, data: bytes, layout: Struct) -> int:
    stream.write(layout.pack(len(data)))
    stream.write(data)
    return layout.size + len(data)


def read_frames(stream: BinaryReader, layout: Struct, *, max_frame_bytes: int | None = None) -> Generator[bytes]:
    while header := stream.read(layout.size):
        if len(header) != layout.size:
            raise ValueError("truncated archive frame header")
        size = layout.unpack(header)[0]
        if max_frame_bytes is not None and size > max_frame_bytes:
            raise ValueError("archive frame exceeds the decoder's frame-size limit")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated archive frame")
        yield data


@dataclass(slots=True)
class BlobSizes:
    stored_bytes: int = 0
    uncompressed_bytes: int | None = 0
    largest_uncompressed_bytes: int | None = 0


def report_file_sizes(path: Path, uncompressed_bytes: int, *, kind: Literal["stream", "members"] = "stream") -> None:
    """Report after publication; callers count raw bytes during encoding/writing.

    Stream sizes include framing and any still-compressed embedded components.
    ZIP member sizes include their .npy headers, not ZIP container overhead.
    Neither measures decoded Python objects or native graph allocations.
    """
    stored = path.stat().st_size
    prefix = "\n" if sys.stderr.isatty() else ""
    print(
        f"{prefix}{path.name}: stored {stored:,} bytes ({stored / 2**20:.2f} MiB); "
        f"uncompressed {kind} {uncompressed_bytes:,} bytes ({uncompressed_bytes / 2**20:.2f} MiB)",
        file=sys.stderr,
    )


def _extend_large_nats(val: object) -> object:
    """Replace integers outside MessagePack's range with canonical extensions."""
    match val:
        case bool() | str() | bytes() | bytearray() | memoryview():
            return val
        case int() as number:
            if number <= _MAX_MSGPACK_NAT:
                return number
            data = number.to_bytes((number.bit_length() + 7) // 8, "big")
            return msgspec.msgpack.Ext(_BIG_NAT_EXT, data)
        case Mapping() as record:
            return {key: _extend_large_nats(item) for key, item in record.items()}
        case Sequence() as items:
            return [_extend_large_nats(item) for item in items]
        case _:
            return val


def decode_nat_ext(code: int, data: memoryview) -> object:
    if code != _BIG_NAT_EXT:
        raise SchemaError(f"unknown MessagePack extension code {code}")
    encoded = bytes(data)
    if not encoded or (len(encoded) > 1 and encoded[0] == 0):
        raise SchemaError("non-canonical arbitrary-precision natural")
    value = int.from_bytes(encoded, "big")
    if value <= _MAX_MSGPACK_NAT:
        raise SchemaError("natural uses an extension despite fitting in MessagePack")
    return value


def encode_msgpack(val: object) -> bytes:
    """Encode records without truncating arbitrary-precision natural numbers."""
    try:
        return _encoder.encode(val)
    except OverflowError:
        # Lean naturals can exceed MessagePack's uint64 range. Most records do
        # not, so only traverse and convert the whole record when necessary.
        return _encoder.encode(
            _extend_large_nats(
                msgspec.to_builtins(val, builtin_types=(bytes, bytearray, memoryview, msgspec.msgpack.Ext, msgspec.Raw))
            )
        )


def data_digest(data: object) -> bytes:
    """Ordinary MessagePack identity, distinct from arbitrary-natural incremental hashing."""
    return hashlib.sha256(msgspec.msgpack.encode(data)).digest()


def canonical_request(data: object) -> object:
    """Canonical request containers, retaining existing JSON/Python equality semantics."""

    def containers(val: object) -> object:
        if isinstance(val, (tuple, list)):
            return [containers(item) for item in val]
        if isinstance(val, dict):
            return {key: containers(item) for key, item in val.items()}
        return val

    return containers(msgspec.to_builtins(data))
