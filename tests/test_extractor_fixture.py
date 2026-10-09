from __future__ import annotations

import csv
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from compression import zstd
from contextlib import closing
from pathlib import Path

from test_extraction_records import stored_theorem

import trustmebro.graph as graph_ops
from trustmebro.extraction import records as r
from trustmebro.extraction.records import SchemaError, iter_theorems
from trustmebro.extraction.scheduler import find_main, read_file_list, sample_files
from trustmebro.extraction.storage import (
    Dicts,
    commit_extracted_file,
    encode_file,
    open_extraction_db,
    recompress_db,
    stored_dicts,
)
from trustmebro.visualization import views

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("src/trustmebro/extraction/Fixture.lean")
LEAN_EXECUTABLE = PROJECT_ROOT / ".lake/build/bin/trustmebro-extract-state"
PRIVATE_IN_PUBLIC = Path("src/trustmebro/extraction/Fixture/Plumbing.lean")


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="surrogateescape", newline="") as stream:
        return list(csv.DictReader(stream))


class ExtractorFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        process = subprocess.run(
            ["lake", "exe", "trustmebro-extract-state", str(FIXTURE)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            timeout=300,
        )
        cls.output = process.stdout
        cls.records = list(iter_theorems(io.BytesIO(process.stdout)))
        cls.transitions = {
            theorem.name.rsplit(".", 1)[-1]: [transition.tactic.src for transition in theorem.trns]
            for theorem in cls.records
        }

    def test_flattened_cached_exports_cross_storage_analysis_and_rendering(self) -> None:
        # Failure cases: pre-cache conversion expands a shared DAG exponentially;
        # alpha equality erases binder annotations; equal bvar syntax is grouped
        # across distinct scopes; safe shifted domains fail to group; wide apps
        # lose ordered/repeated operands or binder multiplicity; ingestion or
        # renamed CLI rejects output.
        # Checks actual Lean -> typed records -> SQLite -> analyses -> PNG, not
        # corpus performance or backward compatibility with old binary records.
        from unittest.mock import patch

        from analysis_fixture import analyze

        from trustmebro.visualization import cli, metrics
        from trustmebro.visualization.products import AnalysisPaths

        records = {item.name.rsplit(".", 1)[-1]: item for item in self.records}
        shared = records["sharedExpressionDag"]
        self.assertLess(len(shared.exprs), 80)
        graph = graph_ops.build_graph(shared.exprs)
        self.assertGreater(graph_ops.graph_stats(graph).sizes[shared.trns[0].state.target], 1 << 32)
        target = shared.exprs[shared.trns[0].state.target]
        self.assertIsInstance(target, r.App)
        assert isinstance(target, r.App)
        self.assertEqual(target.args[-1], target.args[-2])
        unsafe = records["differentlyScopedDomains"].exprs
        self.assertFalse(any(isinstance(expr, r.Forall) and expr.names == ("b", "c") for expr in unsafe))
        self.assertTrue(
            any(
                isinstance(expr, r.Forall)
                and expr.names == ("b",)
                and isinstance(unsafe[expr.body], r.Forall)
                and unsafe[expr.body].names == ("c",)
                for expr in unsafe
            )
        )
        for name, cls in (("sharedOuterDomain", r.Forall), ("sharedLambdaDomain", r.Lambda)):
            self.assertTrue(any(isinstance(expr, cls) and expr.names == ("x", "y") for expr in records[name].exprs))
        names = {expr.names for expr in records["distinctBinderAnnotations"].exprs if isinstance(expr, r.Lambda)}
        self.assertTrue({("left",), ("right",)} <= names)
        wide = records["wideApplication"]
        self.assertTrue(
            any(isinstance(expr, r.App) and len(expr.args) == 128 and len(set(expr.args)) == 1 for expr in wide.exprs)
        )
        wide_graph = graph_ops.build_graph(wide.exprs)
        arrays = graph_ops.expr_arrays(wide_graph)
        self.assertEqual(arrays.children.size, sum(map(len, wide_graph.edges)))
        view = views.prepare_views(wide_graph, (views.ViewMode.ORIGINAL,), matches={})[views.ViewMode.ORIGINAL]
        measured = views.measure_view(
            view,
            graph_ops.graph_stats(wide_graph),
            graph_ops.ReachCache(wide_graph),
            views.view_arrs(view),
            (wide.trns[0].state.target,),
        )
        self.assertGreaterEqual(measured.binders, 128)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "fixture.db"
            with closing(open_extraction_db(db_path)) as db:
                commit_extracted_file(db, FIXTURE, encode_file(io.BytesIO(self.output)))
                self.assertEqual(stored_theorem(db, shared.name), shared)
            # Small batches force motif boundary handling without a large workload.
            with patch.object(metrics, "MOTIF_BATCH_WALKS", 64):
                analyze(db_path, root / "stats", analyses=("metrics", "topology"), depths=(1, 2))
            self.assertTrue(AnalysisPaths(root / "stats").analysis("metrics").is_file())
            self.assertEqual(
                cli.main(["--stats", str(root / "stats"), "--output", str(root / "graphs"), "--graphs", "complexity"]),
                0,
            )
            self.assertTrue(any((root / "graphs").glob("*.png")))

    def test_representation_rewrites_survive_storage_without_losing_argument_boundaries(self) -> None:
        # Failures: operator rewriting guesses operands of partial applications
        # or loops on a bare constant; dependent coercions export the instance
        # instead of their value; function coercions drop trailing arguments or
        # leave an application spine nested; nested argument apps get spliced;
        # alpha-equivalent binders lose their annotations; storage changes IDs.
        # Establishes declared lossy representation rules on kernel-checked
        # source proofs, not semantic equivalence or predictive improvement.
        records = {item.name.rsplit(".", 1)[-1]: item for item in self.records}

        def head_name(name: str, expr: r.App) -> str:
            head = records[name].exprs[expr.fn]
            self.assertIsInstance(head, r.Const)
            assert isinstance(head, r.Const)
            return head.name

        def rhs(name: str) -> r.Expr:
            theorem = records[name]
            goal = theorem.exprs[theorem.trns[0].state.target]
            self.assertIsInstance(goal, r.App)
            assert isinstance(goal, r.App)
            self.assertEqual(head_name(name, goal), "Eq")
            return theorem.exprs[goal.args[-1]]

        for name in ("normalizedNatAdd", "normalizedIntAdd", "normalizedOverloadedAdd"):
            expr = rhs(name)
            self.assertIsInstance(expr, r.App)
            assert isinstance(expr, r.App)
            self.assertEqual(len(expr.args), 2)
            self.assertEqual(records[name].exprs[expr.fn], r.Const("Add.add", ()))
        bare = rhs("bareOperator")
        self.assertIsInstance(bare, r.Const)
        assert isinstance(bare, r.Const)
        self.assertEqual(bare.name, "HAdd.hAdd")
        for name, head, arity in (
            ("partialOperator", "HAdd.hAdd", 2),
            ("partialNativeOperator", "Nat.add", 1),
            ("partialCoercion", "Coe.coe", 2),
        ):
            expr = rhs(name)
            self.assertIsInstance(expr, r.App)
            assert isinstance(expr, r.App)
            self.assertEqual(len(expr.args), arity)
            self.assertEqual(head_name(name, expr), head)
        for name, type_name in (("normalizedNatCast", "Nat"), ("normalizedCoercion", "CoercionBox")):
            expr = rhs(name)
            self.assertIsInstance(expr, r.Fvar)
            assert isinstance(expr, r.Fvar)
            ids = [
                local.id
                for local in records[name].trns[0].state.locals
                if isinstance(type_expr := records[name].exprs[local.type], r.Const)
                and type_expr.name.rsplit(".", 1)[-1] == type_name
            ]
            self.assertEqual(ids, [expr.id])
        dependent = rhs("normalizedDependentCoercion")
        self.assertIsInstance(dependent, r.App)
        assert isinstance(dependent, r.App)
        self.assertEqual(head_name("normalizedDependentCoercion", dependent), "OfNat.ofNat")
        self.assertEqual(records["normalizedDependentCoercion"].exprs[dependent.args[1]], r.NatLiteral(7))

        # Already-unfolded user projections are not guessed to be coercions.
        fn = rhs("normalizedFunctionCoercion")
        self.assertIsInstance(fn, r.App)
        assert isinstance(fn, r.App)
        self.assertEqual(len(fn.args), 2)
        self.assertTrue(head_name("normalizedFunctionCoercion", fn).endswith("FunctionBox.fn"))
        applied = rhs("normalizedAppliedFunctionCoercion")
        self.assertIsInstance(applied, r.App)
        assert isinstance(applied, r.App)
        self.assertEqual(len(applied.args), 2)
        self.assertTrue(head_name("normalizedAppliedFunctionCoercion", applied).endswith("FunctionBox.fn"))
        self.assertIsInstance(records["normalizedAppliedFunctionCoercion"].exprs[applied.args[0]], r.App)
        for name, head in (
            ("rawFunctionCoercion", "FunctionBox.mk"),
            ("rawAppliedFunctionCoercion", "makeFunctionBox"),
        ):
            expr = rhs(name)
            self.assertIsInstance(expr, r.App)
            assert isinstance(expr, r.App)
            self.assertEqual(len(expr.args), 2)
            self.assertTrue(head_name(name, expr).endswith(head))
            operand = records[name].exprs[expr.args[-1]]
            if isinstance(operand, r.App):
                self.assertEqual(head_name(name, operand), "OfNat.ofNat")
                operand = records[name].exprs[operand.args[1]]
            self.assertEqual(operand, r.NatLiteral(3))
        families = {"Add.add", "Mul.mul", "Sub.sub", "Div.div"}
        for name in ("normalizedNatArithmetic", "normalizedFinArithmetic"):
            observed = {
                head_name(name, expr)
                for expr in records[name].exprs
                if isinstance(expr, r.App)
                and isinstance(records[name].exprs[expr.fn], r.Const)
                and head_name(name, expr) in families
                and len(expr.args) == 2
            }
            self.assertEqual(observed, families)
        nested = rhs("nestedApplications")
        self.assertIsInstance(nested, r.App)
        assert isinstance(nested, r.App)
        self.assertEqual(len(nested.args), 4)
        argument = records["nestedApplications"].exprs[nested.args[1]]
        self.assertIsInstance(argument, r.App)
        assert isinstance(argument, r.App)
        self.assertEqual(len(argument.args), 4)
        binders = {expr.binder_info for expr in records["distinctBinderKinds"].exprs if isinstance(expr, r.Lambda)}
        self.assertEqual(binders, {r.BinderInfo.DEFAULT, r.BinderInfo.IMPLICIT})

        with tempfile.TemporaryDirectory() as tmp, closing(open_extraction_db(Path(tmp) / "fixture.db")) as db:
            commit_extracted_file(db, FIXTURE, encode_file(io.BytesIO(self.output)))
            for theorem in self.records:
                self.assertEqual(stored_theorem(db, theorem.name), theorem)

    def test_file_list_runs_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            database = Path(directory, "output.sqlite")
            file_list.write_text(
                f"# missing/Skipped.lean\n{FIXTURE}\n{PRIVATE_IN_PUBLIC}\nsrc/trustmebro/extraction/Extract.lean\n"
            )
            original_list = file_list.read_text()
            command = [
                str(Path(sys.executable).parent / "proof-states"),
                "--workers",
                "2",
                "--files-from",
                str(file_list),
                "--db",
                str(database),
                "--lean-exe",
                str(LEAN_EXECUTABLE),
            ]
            process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, timeout=300, check=False)
            self.assertEqual(process.returncode, 0, process.stderr.decode())
            summary = json.loads(process.stdout)
            self.assertEqual(summary["theorems"], len(self.records) + 10)
            with closing(sqlite3.connect(database)) as connection:
                # Failures: CLI omits/mislabels component sizes, counts SQLite
                # overhead as blobs, or reports headers inconsistent with the
                # actual uncompressed MessagePack. This fixture uses plain Zstd.
                for col in ("exprs", "trns"):
                    blobs = [row[0] for row in connection.execute(f"SELECT {col} FROM theorems")]
                    raw_sizes = [len(zstd.decompress(blob)) for blob in blobs]
                    self.assertEqual(
                        summary["blob_sizes"][col],
                        {
                            "stored_bytes": sum(map(len, blobs)),
                            "uncompressed_bytes": sum(raw_sizes),
                            "largest_uncompressed_bytes": max(raw_sizes),
                        },
                    )
                count = connection.execute("SELECT COUNT(*) FROM theorems").fetchone()[0]
                self.assertEqual(count, summary["theorems"])
                completed = connection.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0]
                self.assertEqual(completed, 3)
                self.assertEqual(
                    connection.execute(
                        "SELECT theorem_count, expr_count, trn_count FROM completed_files WHERE src_path = ?",
                        (str(Path("src/trustmebro/extraction/Extract.lean").resolve()),),
                    ).fetchone(),
                    (0, 0, 0),
                )
                source_paths = {row[0] for row in connection.execute("SELECT DISTINCT src_path FROM theorems")}
                self.assertEqual(source_paths, {str(FIXTURE.resolve()), str(PRIVATE_IN_PUBLIC.resolve())})
            self.assertEqual(file_list.read_text(), original_list)
            self.assertNotIn(b"Exported", process.stderr)
            logs = read_events(database.with_suffix(".log.csv"))
            self.assertEqual(len(logs), 3)
            self.assertTrue(all(record["status"] == "completed" for record in logs))
            self.assertTrue(all(float(record["dur_sec"]) >= 0 for record in logs))
            self.assertTrue(all(record["diagns"] == "" for record in logs))
            self.assertTrue(all(record["setup_ms"] == "" for record in logs))
            self.assertEqual(sum(int(record["exprs"]) for record in logs), summary["exprs"])

            repeated = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, check=True, timeout=30)
            self.assertEqual(
                {k: v for k, v in json.loads(repeated.stdout).items() if k != "dur_sec"},
                {k: v for k, v in summary.items() if k != "dur_sec"},
            )
            self.assertEqual(file_list.read_text(), original_list)
            self.assertEqual(len(read_events(database.with_suffix(".log.csv"))), 3)

    def test_worker_limit_must_be_positive(self) -> None:
        process = subprocess.run(
            [
                str(Path(sys.executable).parent / "proof-states"),
                "--workers",
                "0",
                "--files-from",
                "unused",
                "--db",
                "unused",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"must be a positive integer", process.stderr)

    def test_commented_existing_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            database = Path(directory, "output.sqlite")
            file_list.write_text(f"# {PRIVATE_IN_PUBLIC}\n{FIXTURE}\n")
            process = subprocess.run(
                [
                    str(Path(sys.executable).parent / "proof-states"),
                    "--files-from",
                    str(file_list),
                    "--db",
                    str(database),
                    "--lean-exe",
                    str(LEAN_EXECUTABLE),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=True,
                timeout=300,
            )
            self.assertEqual(json.loads(process.stdout)["theorems"], len(self.records))
            self.assertEqual(file_list.read_text().splitlines(), [f"# {PRIVATE_IN_PUBLIC}", str(FIXTURE)])

    def test_failed_file_preserves_list_and_previous_file_is_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            database = Path(directory, "output.sqlite")
            bad_file = Path(directory, "OutsideLake.lean")
            bad_file.write_text("theorem foo : True := by trivial\n")
            file_list.write_text(f"{FIXTURE}\n{bad_file}\n")
            process = subprocess.run(
                [
                    str(Path(sys.executable).parent / "proof-states"),
                    "--workers",
                    "1",
                    "--files-from",
                    str(file_list),
                    "--db",
                    str(database),
                    "--lean-exe",
                    str(LEAN_EXECUTABLE),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                timeout=300,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(file_list.read_text().splitlines(), [str(FIXTURE), str(bad_file)], process.stderr.decode())
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0], 1)
            logs = read_events(database.with_suffix(".log.csv"))
            self.assertEqual([record["status"] for record in logs], ["completed", "failed"])
            self.assertIn(str(bad_file), logs[-1]["src"])

    def test_timing_is_recorded_in_per_file_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            database = Path(directory, "output.sqlite")
            log_file = Path(directory, "extract.csv")
            file_list.write_text(f"{FIXTURE}\n")
            process = subprocess.run(
                [
                    str(Path(sys.executable).parent / "proof-states"),
                    "--files-from",
                    str(file_list),
                    "--db",
                    str(database),
                    "--log-file",
                    str(log_file),
                    "--timing",
                    "--lean-exe",
                    str(LEAN_EXECUTABLE),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                timeout=300,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr.decode())
            (record,) = read_events(log_file)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(int(record["theorems"]), len(self.records))
            self.assertGreaterEqual(int(record["setup_ms"]), 0)
            self.assertGreaterEqual(int(record["elaboration_ms"]), 0)
            self.assertGreaterEqual(int(record["json_build_ms"]), 0)
            self.assertGreaterEqual(int(record["json_encode_ms"]), 0)
            self.assertNotIn("stdout_write_ms", record)
            self.assertNotIn("read_parse_ms", record)
            self.assertNotIn("serialization_ms", record)
            self.assertEqual(record["diagns"], "")
            self.assertNotIn(b"TIMING", process.stderr)

    def test_worker_stderr_bytes_survive_csv_logging(self) -> None:
        # Lean normally emits UTF-8, but replacement decoding would silently
        # alter unexpected diagnostic bytes before they reach the log.
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            file_list.write_text(f"{FIXTURE}\n")
            database = Path(directory, "output.sqlite")
            executable = Path(directory, "failed-worker")
            executable.write_text(
                f"#!{sys.executable}\nimport sys\nsys.stderr.buffer.write(b'failure: \\xff\\n')\nsys.exit(1)\n"
            )
            executable.chmod(0o755)
            process = subprocess.run(
                [
                    str(Path(sys.executable).parent / "proof-states"),
                    "--files-from",
                    str(file_list),
                    "--db",
                    str(database),
                    "--lean-exe",
                    str(executable),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            records = read_events(database.with_suffix(".log.csv"))
            self.assertEqual(len(records), 1, process.stderr.decode())
            (record,) = records
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["diagns"].encode("utf-8", errors="surrogateescape"), b"failure: \xff\n")
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0], 0)

    def test_mathlib_private_declaration_elaborates(self) -> None:
        process = subprocess.run(
            ["lake", "exe", "trustmebro-extract-state", str(PRIVATE_IN_PUBLIC)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            timeout=300,
        )
        records = list(iter_theorems(io.BytesIO(process.stdout)))
        self.assertEqual(len(records), 10)
        self.assertGreater(sum(len(item.trns) for item in records), 0)

    def test_lean_to_sqlite_and_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory, "files.txt")
            file_list.write_text(f"{FIXTURE}\n")
            database = Path(directory, "fixture.sqlite")
            command = [
                str(Path(sys.executable).parent / "proof-states"),
                "--files-from",
                str(file_list),
                "--db",
                str(database),
                "--lean-exe",
                str(LEAN_EXECUTABLE),
            ]
            process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, check=True, timeout=300)
            summary = json.loads(process.stdout)
            self.assertEqual(summary["theorems"], len(self.records))
            self.assertEqual(summary["trns"], sum(len(item.trns) for item in self.records))
            self.assertEqual(summary["exprs"], sum(len(item.exprs) for item in self.records))

            with closing(sqlite3.connect(database)) as connection:
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
                self.assertEqual(set(tables), {("theorems",), ("completed_files",), ("compression_dicts",)})
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM theorems WHERE src_path = ?", (str(FIXTURE.resolve()),)
                    ).fetchone()[0],
                    len(self.records),
                )
                for theorem in self.records:
                    with self.subTest(theorem=theorem.name):
                        self.assertEqual(stored_theorem(connection, theorem.name), theorem)

            again = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, check=True, timeout=30)
            self.assertEqual(
                {k: v for k, v in json.loads(again.stdout).items() if k != "dur_sec"},
                {k: v for k, v in summary.items() if k != "dur_sec"},
            )

    def test_discovery_accepts_nested_tactic_terms_and_ignores_noncode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snippets = {
                "Nested": "theorem proof : True ∧ True := ⟨by trivial, by trivial⟩",
                "Lambda": "theorem proof : True → True := fun h => by exact h",
                "Comments": "/- theorem /- lemma by -/ by -/ -- example by\n",
                "Literals": 'def x := r###"theorem " by"###\ndef y := "example by"',
                "Identifiers": "def «theorem» := 0\ndef αby := 0\ndef by' := 0\ndef example_name := 0",
            }
            for name, source in snippets.items():
                (root / f"{name}.lean").write_text(source)
            output = root / "candidates.txt"
            process = subprocess.run(
                [str(Path(sys.executable).parent / "find-files"), "--root", str(root), "--output", str(output)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                {Path(line).stem for line in output.read_text().splitlines()},
                {"Nested", "Lambda"},
                process.stderr.decode(),
            )

    def test_discovery_sampling_and_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            (sources / "A.lean").symlink_to(FIXTURE.resolve())
            (sources / "B.lean").symlink_to(PRIVATE_IN_PUBLIC.resolve())
            (sources / "Comments.lean").write_text(
                "/- theorem outer /- example nested := by -/ lemma ignored -/\n"
                "-- theorem x := by trivial\n"
                'def text := "lemma x := by trivial"\n'
                'def raw := r##"theorem x " by trivial"##\n'
                "def «by» := 1\n"
                "def «theorem» := 1\n"
                "def theorem_name := 1\n"
                "def example' := 'x'\n"
                'def text2 := "escaped \\" lemma by"\n'
            )
            candidates = root / "candidates.txt"
            selection = root / "sample.txt"
            self.assertEqual(find_main(["--root", str(sources), "--output", str(candidates)]), 0)
            self.assertEqual(
                set(candidates.read_text().splitlines()), {str(FIXTURE.resolve()), str(PRIVATE_IN_PUBLIC.resolve())}
            )
            selection.write_text("\n".join(map(str, sample_files(read_file_list(candidates), 1, 2))) + "\n")
            self.assertEqual(selection.read_text().strip(), str(FIXTURE.resolve()))
            database = root / "full.sqlite"
            command = [
                str(Path(sys.executable).parent / "extract-dataset"),
                "--root",
                str(sources),
                "--db",
                str(database),
                "--sample-size",
                "1",
                "--seed",
                "2",
                "--dictionary-size",
                "512",
                "--workers",
                "2",
                "--lean-exe",
                str(LEAN_EXECUTABLE),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, timeout=300, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            summary = json.loads(result.stdout)
            self.assertEqual(summary["theorems"], len(self.records) + 10)
            self.assertGreater(summary["dur_sec"], 0)
            self.assertEqual(summary["db"], str(database.resolve()))
            self.assertEqual(summary["db_bytes"], database.stat().st_size)
            for stage in (
                "Discover candidate files",
                "Select initial sample",
                "Extract initial sample",
                "Sample: train dictionaries",
                "Sample: recompress database",
                "Extract remaining files",
                "Full corpus: train dictionaries",
                "Full corpus: recompress database",
            ):
                self.assertIn(f"{stage} completed in ".encode(), result.stderr)
            self.assertNotIn(b"\r", result.stderr)
            self.assertIn(b"expression bytes", result.stderr)
            self.assertIn(b"MiB", result.stderr)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT phase FROM extraction_pipeline").fetchone(), ("done",))
                self.assertEqual(connection.execute("SELECT count(*) FROM compression_dicts").fetchone()[0], 2)
                for theorem in self.records:
                    self.assertEqual(stored_theorem(connection, theorem.name), theorem)
            before = database.read_bytes()
            resumed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, check=True)
            self.assertEqual(
                {k: v for k, v in json.loads(resumed.stdout).items() if k != "dur_sec"},
                {k: v for k, v in summary.items() if k != "dur_sec"},
            )
            self.assertEqual(database.read_bytes(), before)

            # Faults valid Lean cannot produce: unreadable compressed input,
            # or a database error while replacing dictionaries after blob updates.
            # Both must roll back earlier changes and retain the old dictionaries.
            with closing(sqlite3.connect(database, isolation_level=None)) as connection:
                original = connection.execute("SELECT id, exprs, trns FROM theorems ORDER BY id").fetchall()
                dictionaries = stored_dicts(connection)
                assert dictionaries is not None
                replacement = Dicts(dictionaries.trns, dictionaries.exprs)
                connection.execute(
                    "CREATE TRIGGER reject_dictionary BEFORE INSERT ON compression_dicts "
                    "BEGIN SELECT RAISE(ABORT, 'injected dictionary write failure'); END"
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    recompress_db(connection, replacement)
                self.assertEqual(
                    connection.execute("SELECT id, exprs, trns FROM theorems ORDER BY id").fetchall(), original
                )
                self.assertEqual(stored_dicts(connection), dictionaries)
                connection.execute("DROP TRIGGER reject_dictionary")
                connection.execute("UPDATE theorems SET exprs=? WHERE id=?", (b"broken frame", original[1][0]))
                with self.assertRaises(SchemaError):
                    recompress_db(connection, replacement)
                self.assertEqual(
                    connection.execute("SELECT exprs FROM theorems WHERE id=?", (original[0][0],)).fetchone()[0],
                    original[0][1],
                )
                self.assertEqual(stored_dicts(connection), dictionaries)

    def test_full_pipeline_resumes_failed_remaining_file(self) -> None:
        # Inject a worker failure after sample training/recompression; resume
        # must preserve the sample, extract only the missing file, and finalize.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            for name in ("A", "B"):
                (sources / f"{name}.lean").write_text("theorem t : True := by trivial\n")
            output = root / "fixture.ndjson"
            output.write_bytes(self.output)
            failure = root / "fail"
            failure.touch()
            executable = root / "worker"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json,sys\nfrom pathlib import Path\n"
                "name=Path(sys.argv[1]).stem\n"
                f"if name == 'B' and Path({str(failure)!r}).exists(): sys.exit(1)\n"
                f"for line in Path({str(output)!r}).read_text().splitlines():\n"
                " record=json.loads(line)\n"
                " record['name']=name+'.'+record['name']\n"
                " print(json.dumps(record))\n"
            )
            executable.chmod(0o755)
            database = root / "full.sqlite"
            command = [
                str(Path(sys.executable).parent / "extract-dataset"),
                "--root",
                str(sources),
                "--db",
                str(database),
                "--sample-size",
                "1",
                "--dictionary-size",
                "512",
                "--lean-exe",
                str(executable),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, timeout=60, check=False)
            self.assertNotEqual(result.returncode, 0)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT phase FROM extraction_pipeline").fetchone(), ("rest",))
                self.assertEqual(connection.execute("SELECT count(*) FROM completed_files").fetchone()[0], 1)
                for theorem in self.records:
                    self.assertEqual(stored_theorem(connection, "A." + theorem.name).exprs, theorem.exprs)
            failure.unlink()
            resumed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, timeout=60, check=False)
            self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
            self.assertEqual(json.loads(resumed.stdout)["theorems"], 2 * len(self.records))
            logs = read_events(database.with_suffix(".log.csv"))
            self.assertEqual([r["status"] for r in logs], ["completed", "failed", "completed"])
            changed = subprocess.run([*command, "--seed", "3"], cwd=PROJECT_ROOT, capture_output=True, check=False)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn(b"configuration", changed.stderr)

    def test_macro_expansion_retains_only_callable_source_tactic(self) -> None:
        self.assertEqual(self.transitions["sourceMacro"], ["fixture_exact hp"])

    def test_first_selects_successful_branches_across_implementation_kinds(self) -> None:
        expected = {
            "firstMacro": ["fixture_exact hp"],
            "firstElaborator": ["fixture_elab_assumption"],
            "firstRing": ["ring"],
            "firstOmega": ["omega"],
            "firstAutomationFallback": ["fixture_exact hp"],
        }
        for theorem, transitions in expected.items():
            with self.subTest(theorem=theorem):
                self.assertEqual(self.transitions[theorem], transitions)

    def test_rolled_back_partial_progress_is_not_exported(self) -> None:
        expected = {
            "tryMacroRollback": ["exact ⟨hp, hq⟩"],
            "tryElaboratorRollback": ["exact ⟨hp, hq⟩"],
            "solveRollback": ["exact ⟨hp, hq⟩"],
            "repeatMacroRollback": ["exact ⟨hp, hq⟩"],
            "repeatElaboratorRollback": ["exact ⟨hp, hq⟩"],
            "anyGoalsMacroRollback": ["constructor", "exact True.intro", "exact ⟨hp, hq⟩"],
            "anyGoalsElaboratorRollback": ["constructor", "exact True.intro", "exact ⟨hp, hq⟩"],
            "anyGoalsAutomationRollback": ["constructor", "exact True.intro", "ring"],
        }
        for theorem, transitions in expected.items():
            with self.subTest(theorem=theorem):
                self.assertEqual(self.transitions[theorem], transitions)


if __name__ == "__main__":
    unittest.main()
