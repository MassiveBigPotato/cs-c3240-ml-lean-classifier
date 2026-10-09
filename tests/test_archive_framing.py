"""Archive adapters retain their distinct wire layouts and truncation checks."""

from compression import zstd
from pathlib import Path
from struct import Struct
from tempfile import TemporaryDirectory
from unittest import TestCase

import msgspec

from trustmebro.artifacts import encode_msgpack
from trustmebro.preprocessing import archives as pre
from trustmebro.visualization import archives as viz


class ArchiveFramingTests(TestCase):
    def test_archive_writers_preserve_wire_formats_and_exact_records(self) -> None:
        # Failures: shared framing changes byte order, loses large naturals,
        # miscounts uncompressed bytes, stringifies binary fallback fields,
        # or changes EOF handling. Establishes
        # byte-level archive compatibility, not full plotting integration.
        item = {"count": 1 << 100, "refs": [1, 1, 0], "binary": memoryview(b"\x00\xff"), "mutable": bytearray(b"abc")}
        encoded = encode_msgpack(item)
        self.assertEqual(
            encoded,
            msgspec.msgpack.encode(
                {
                    "count": msgspec.msgpack.Ext(1, (1 << 100).to_bytes(13, "big")),
                    "refs": [1, 1, 0],
                    "binary": b"\x00\xff",
                    "mutable": b"abc",
                }
            ),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pre_path, viz_path = root / "pre.zst", root / "viz.zst"
            raw_bytes = pre.publish(pre_path, [encoded], sources=(), replace=False)
            viz.write_summary(viz_path, item)
            for path, layout, read in (
                (pre_path, Struct("!Q"), pre.read_frames),
                (viz_path, Struct("<Q"), viz.records),
            ):
                with self.subTest(path=path.name):
                    with zstd.open(path, "rb") as stream:
                        self.assertEqual(stream.read(), layout.pack(len(encoded)) + encoded)
                    self.assertEqual(list(read(path)), [encoded])
                    self.assertEqual(viz.decode_record(encoded), item)
            self.assertEqual(raw_bytes, 8 + len(encoded))

    def test_archive_readers_reject_truncation_and_preprocessing_frame_cap(self) -> None:
        # Failures: short headers/bodies silently terminate; the cap is checked
        # only after allocating/reading the body. Tiny malformed public archives
        # exercise both adapters without full measurement or candidate fixtures.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.zst"
            for layout, read in ((Struct("!Q"), pre.read_frames), (Struct("<Q"), viz.records)):
                for data in (b"\x01", layout.pack(5) + b"abc"):
                    with self.subTest(layout=layout.format, data=data):
                        with zstd.open(path, "wb") as stream:
                            stream.write(data)
                        with self.assertRaisesRegex(ValueError, "truncated"):
                            list(read(path))
            with zstd.open(path, "wb") as stream:
                stream.write(Struct("!Q").pack(100))  # no body: limit must fail first
            with self.assertRaisesRegex(ValueError, "frame-size limit"):
                list(pre.read_frames(path, max_frame_bytes=10))
