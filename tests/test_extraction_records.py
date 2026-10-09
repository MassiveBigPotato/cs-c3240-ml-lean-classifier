from __future__ import annotations

import copy
import io
import json
import sqlite3
import tempfile
import unittest
from compression import zstd
from contextlib import closing
from pathlib import Path

import msgspec

from trustmebro.extraction.records import Let, LvlIMax, Metadata, Proj, SchemaError, Sort, iter_theorems
from trustmebro.extraction.storage import (
    THEOREM_COLS,
    BlobCodec,
    commit_extracted_file,
    decode_theorem_row,
    encode_file,
    open_extraction_db,
    stored_dicts,
)


def stored_theorem(db: sqlite3.Connection, name: str):
    """Decode fixture rows through the same boundary used by retained consumers."""
    row = db.execute(f"SELECT {THEOREM_COLS} FROM theorems WHERE name = ?", (name,)).fetchone()
    return decode_theorem_row(row, BlobCodec(stored_dicts(db)))


def store_record(record: dict[str, object], database: Path) -> None:
    stream = io.BytesIO(json.dumps(record).encode() + b"\n")
    encoded = encode_file(stream)
    with closing(open_extraction_db(database)) as connection:
        commit_extracted_file(connection, Path("Fixture.lean"), encoded)


def theorem_record() -> dict[str, object]:
    return {
        "name": "Fixture.example",
        "module": "Fixture",
        "srcSpan": [10, 80],
        "exprs": [
            {"const": {"name": "Nat", "universes": []}},
            {"bvar": 0},
            {"forall": {"names": ["n"], "type": 0, "body": 1, "binderInfo": "default"}},
            {"app": {"fn": 0, "args": [1]}},
        ],
        "trns": [
            {
                "tactic": {"kind": "Lean.Parser.Tactic.exact", "src": "exact n"},
                "srcSpan": [50, 57],
                "openGoalCount": 1,
                "state": {
                    "target": 3,
                    "locals": [{"id": 0, "type": 0, "kind": "default", "isInstance": False, "binderInfo": "default"}],
                    "mvars": [],
                    "lvlMvars": [],
                },
            }
        ],
    }


def all_variants_record() -> dict[str, object]:
    record = theorem_record()
    record["exprs"] = [
        {"bvar": 0},
        {"fvar": 0},
        {"mvar": 0},
        {"sort": "zero"},
        {
            "const": {
                "name": "Fixture.constant",
                "universes": [
                    {"mvar": 0},
                    {"succ": {"param": "u"}},
                    {"max": ["zero", {"param": "v"}]},
                    {"imax": [{"param": "u"}, {"param": "v"}]},
                ],
            }
        },
        {"app": {"fn": 4, "args": [1]}},
        {"lambda": {"names": ["x"], "type": 3, "body": 5, "binderInfo": "implicit"}},
        {"forall": {"names": ["x"], "type": 3, "body": 6, "binderInfo": "strictImplicit"}},
        {"let": {"name": "x", "type": 3, "val": 4, "body": 7, "nondep": False}},
        {"natural": 184467440737095516160},
        {"string": "value"},
        {"metadata": {"data": {"key": "value"}, "expr": 10}},
        {"projection": {"typeName": "Fixture.Structure", "idx": 0, "struct": 11}},
    ]
    transitions = record["trns"]
    assert isinstance(transitions, list)
    transition = transitions[0]
    assert isinstance(transition, dict)
    state = transition["state"]
    assert isinstance(state, dict)
    state.update(
        {
            "target": 12,
            "locals": [
                {"id": 0, "type": 3, "kind": "implementationDetail", "isInstance": True, "binderInfo": "instImplicit"},
                {"id": 1, "type": 3, "kind": "auxiliary", "isInstance": False, "val": 9, "nondep": True},
            ],
            "mvars": [
                {
                    "id": 0,
                    "decl": {"type": 12, "locals": [], "kind": "syntheticOpaque"},
                    "assignment": 11,
                    "delayedAssign": {"fvars": [1], "pending": 1},
                }
            ],
            "lvlMvars": [{"id": 0, "decl": {"depth": 1, "idx": 2}, "assignment": {"imax": ["zero", {"param": "u"}]}}],
        }
    )
    return record


class RecordTests(unittest.TestCase):
    def test_variants_and_large_natural_not_in_lean_fixture(self) -> None:
        # The small Lean fixture cannot reliably produce every expression shape
        # or a natural number beyond MessagePack's native integer range.
        record = all_variants_record()
        theorem = next(iter_theorems(io.BytesIO(json.dumps(record).encode() + b"\n")))

        self.assertIsInstance(theorem.exprs[3], Sort)
        self.assertIsInstance(theorem.exprs[8], Let)
        self.assertIsInstance(theorem.exprs[11], Metadata)
        self.assertIsInstance(theorem.exprs[12], Proj)
        self.assertIsInstance(theorem.trns[0].state.lvl_mvars[0].assignment, LvlIMax)
        self.assertEqual(theorem.exprs[9].val, 184467440737095516160)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, "variants.sqlite")
            store_record(record, database)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(stored_theorem(connection, theorem.name), theorem)

    def test_malformed_lean_json_is_rejected(self) -> None:
        # Lean should never emit these, so a successful end-to-end extraction
        # cannot exercise the parser's rejection paths. Empty operand/binder
        # groups must not slip through when container annotations change.
        cases = []

        record = theorem_record()
        record["extra"] = True
        cases.append(("unknown top-level field", record, "unknown field"))

        record = theorem_record()
        record["exprs"][3]["app"]["extra"] = 2
        cases.append(("unknown nested field", record, "unknown field"))

        record = theorem_record()
        record["exprs"][3]["app"]["fn"] = True
        cases.append(("wrong nested type", record, "Expected .*int"))

        record = theorem_record()
        record["exprs"][3]["app"]["args"] = []
        cases.append(("empty application", record, "length >= 1"))

        record = theorem_record()
        record["exprs"][3] = {"lambda": {"names": [], "type": 0, "body": 1, "binderInfo": "default"}}
        cases.append(("empty binder group", record, "length >= 1"))

        record = theorem_record()
        record["exprs"][3]["app"]["fn"] = 4
        cases.append(("forward expression reference", record, "not earlier"))

        record = theorem_record()
        record["exprs"][1] = {"natural": -1}
        cases.append(("negative natural", record, "int.*>= 0"))

        record = theorem_record()
        record["trns"] = []
        cases.append(("missing transitions", record, "length >= 1"))

        record = theorem_record()
        record["srcSpan"] = [80, 10]
        cases.append(("reversed source span", record, "stop offset precedes"))

        for descr, record, message in cases:
            with self.subTest(case=descr), self.assertRaisesRegex(SchemaError, message):
                next(iter_theorems(io.BytesIO(json.dumps(record).encode() + b"\n")))

    def test_corrupt_messagepack_is_rejected(self) -> None:
        # Valid encoder output cannot contain these corrupt tags, references,
        # shapes, or extension values.
        record = theorem_record()
        name = record["name"]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, "corrupt-msgpack.sqlite")
            store_record(record, database)
            with closing(sqlite3.connect(database)) as connection:
                expr_blob, step_blob = connection.execute(
                    "SELECT exprs, trns FROM theorems WHERE name = ?", (name,)
                ).fetchone()
                expressions = msgspec.msgpack.decode(zstd.decompress(expr_blob))
                transitions = msgspec.msgpack.decode(zstd.decompress(step_blob))
                cases = []

                corrupted = copy.deepcopy(expressions)
                corrupted[0][0] = 99
                cases.append(("unknown expression tag", "exprs", corrupted, None))

                corrupted = copy.deepcopy(expressions)
                corrupted[3][1] = 4
                cases.append(("forward expression reference", "exprs", corrupted, "not earlier"))

                corrupted = copy.deepcopy(transitions)
                corrupted[0][2] = 0
                cases.append(("no active goal", "trns", corrupted, None))

                corrupted = copy.deepcopy(expressions)
                corrupted[0].append("unexpected")
                cases.append(("extra expression field", "exprs", corrupted, None))

                for descr, col, corrupted, message in cases:
                    with self.subTest(case=descr):
                        connection.execute(
                            f"UPDATE theorems SET {col} = ? WHERE name = ?",
                            (zstd.compress(msgspec.msgpack.encode(corrupted)), name),
                        )
                        if message is None:
                            with self.assertRaises(SchemaError):
                                stored_theorem(connection, name)
                        else:
                            with self.assertRaisesRegex(SchemaError, message):
                                stored_theorem(connection, name)
                        connection.execute(
                            f"UPDATE theorems SET {col} = ? WHERE name = ?",
                            (expr_blob if col == "exprs" else step_blob, name),
                        )

            large_record = all_variants_record()
            large_name = large_record["name"]
            large_database = Path(directory, "large-natural.sqlite")
            store_record(large_record, large_database)
            with closing(sqlite3.connect(large_database)) as connection:
                (large_blob,) = connection.execute(
                    "SELECT exprs FROM theorems WHERE name = ?", (large_name,)
                ).fetchone()
                corrupted = msgspec.msgpack.decode(zstd.decompress(large_blob))
                corrupted[9][1] = msgspec.msgpack.Ext(1, b"\x00\x01")
                connection.execute(
                    "UPDATE theorems SET exprs = ? WHERE name = ?",
                    (zstd.compress(msgspec.msgpack.encode(corrupted)), large_name),
                )
                with self.assertRaisesRegex(SchemaError, "non-canonical"):
                    stored_theorem(connection, large_name)

    def test_duplicate_theorem_cannot_mark_a_file_complete(self) -> None:
        # Failure modes absent from the Lean fixture: duplicate names within
        # one stream, and a name already committed by an earlier source file.
        line = json.dumps(theorem_record()).encode() + b"\n"
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, "duplicate.sqlite")
            with closing(open_extraction_db(database)) as connection:
                with self.assertRaisesRegex(SchemaError, "duplicate theorem name"):
                    encode_file(io.BytesIO(line + line))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0], 0)

                encoded = encode_file(io.BytesIO(line))
                commit_extracted_file(connection, Path("first.lean"), encoded)
                with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE constraint"):
                    commit_extracted_file(connection, Path("second.lean"), encoded)
                self.assertEqual(
                    connection.execute("SELECT src_path FROM completed_files").fetchall(), [("first.lean",)]
                )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM theorems").fetchone()[0], 1)

    def test_corrupt_stored_counts_and_blobs_are_rejected(self) -> None:
        # These faults cannot arise from valid Lean output or the writer, but
        # retained row decoders must not silently return incorrect data.
        record = theorem_record()
        name = record["name"]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, "corrupt.sqlite")
            store_record(record, database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("UPDATE theorems SET expr_count = expr_count + 1")
                with self.assertRaisesRegex(SchemaError, "expression count"):
                    stored_theorem(connection, name)

                connection.execute("UPDATE theorems SET expr_count = expr_count - 1, trns = ?", (b"\xc1",))
                with self.assertRaises(SchemaError):
                    stored_theorem(connection, name)


if __name__ == "__main__":
    unittest.main()
