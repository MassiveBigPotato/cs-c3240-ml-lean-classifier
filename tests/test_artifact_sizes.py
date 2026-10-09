"""Archive publication and reporting, without corpus analysis or rendering."""

import io
import sqlite3
import tempfile
import unittest
from compression import zstd
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction.storage import db_blob_sizes
from trustmebro.visualization import archives


class ArtifactSizeTests(unittest.TestCase):
    def test_database_component_sizes_use_headers_without_decoding(self) -> None:
        # Failures: dictionary compression prevents header-only sizing; blob
        # totals/maxima disagree; missing declared sizes become zero/estimates;
        # inspection mutates the DB or activates full-record decompression.
        # This header-only resource boundary cannot be established from identical
        # CLI output alone. It does not validate compressed payload integrity.
        dictionary = zstd.train_dict([bytes([idx]) * 200 + b"expression" * 80 for idx in range(32)], 512)
        exprs, trns = (b"expression" * 40, b"expression" * 100), (b"transition" * 90, b"transition" * 20)
        blobs = tuple(
            (zstd.compress(expr, zstd_dict=dictionary), zstd.compress(trn, zstd_dict=dictionary))
            for expr, trn in zip(exprs, trns, strict=True)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.db"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE theorems(exprs BLOB, trns BLOB)")
                db.executemany("INSERT INTO theorems VALUES (?,?)", blobs)
                db.commit()
            original = path.read_bytes()
            with (
                closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db,
                patch.object(zstd, "decompress", side_effect=AssertionError("unexpected decompression")),
            ):
                measured = db_blob_sizes(db)
            for idx, (col, raw) in enumerate((("exprs", exprs), ("trns", trns))):
                self.assertEqual(measured[col].stored_bytes, sum(len(row[idx]) for row in blobs))
                self.assertEqual(measured[col].uncompressed_bytes, sum(map(len, raw)))
                self.assertEqual(measured[col].largest_uncompressed_bytes, max(map(len, raw)))
            self.assertEqual(path.read_bytes(), original)
            unknown = io.BytesIO()
            with zstd.open(unknown, "wb") as stream:
                stream.write(b"unknown-length stream")
            with closing(sqlite3.connect(path)) as db:
                db.execute("INSERT INTO theorems VALUES (?,?)", (blobs[0][0], unknown.getvalue()))
                db.commit()
                measured = db_blob_sizes(db)
            self.assertIsNone(measured["trns"].uncompressed_bytes)
            self.assertIsNone(measured["trns"].largest_uncompressed_bytes)

    def test_archive_reports_exact_framed_stream_bytes_after_publication(self) -> None:
        # Failures: compressed size mistaken for raw size; framing omitted;
        # oversized integers miscounted; reporting leaks into machine stdout;
        # size computation alters records or requires another decoding pass.
        # Checks encoded stream bytes, not nested blob or decoded object sizes.
        values = (("natural", 1 << 15000), ("nested", zstd.compress(b"example" * 100)))
        encoded = tuple(map(encode_msgpack, values))
        expected = b"".join(archives.LENGTH.pack(len(data)) + data for data in encoded)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.msgpack.zst"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), archives.writer(path) as write:
                for val in values:
                    write(val)
            self.assertEqual(zstd.decompress(path.read_bytes()), expected)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(f"stored {path.stat().st_size:,} bytes", stderr.getvalue())
            self.assertIn(f"uncompressed stream {len(expected):,} bytes", stderr.getvalue())
            self.assertEqual(len(stderr.getvalue().splitlines()), 1)

    def test_failed_and_temporary_writes_do_not_report_published_sizes(self) -> None:
        # Failures: a failed replacement reports success or destroys the old
        # archive; temporary spills spam size output; partial files survive.
        # Checks archive reporting/publication boundaries, not pipeline progress.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.msgpack.zst"
            stderr = io.StringIO()
            with redirect_stderr(stderr), archives.writer(path, report_sizes=False) as write:
                write(("original", 1))
            self.assertEqual(stderr.getvalue(), "")
            original = path.read_bytes()
            with (
                self.assertRaisesRegex(RuntimeError, "producer failed"),
                redirect_stderr(stderr),
                archives.writer(path) as write,
            ):
                write(("replacement", 2))
                raise RuntimeError("producer failed")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
