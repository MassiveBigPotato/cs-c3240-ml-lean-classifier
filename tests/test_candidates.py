"""Behavioral DB-to-candidate coverage; no Mathlib scan or performance claims."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections import Counter, deque
from collections.abc import Iterator
from compression import zstd
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path
from threading import Event
from unittest.mock import patch

import msgspec

import trustmebro.graph as graph_ops
from trustmebro import runtime
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.preprocessing import archives, records
from trustmebro.preprocessing import scan as scheduler


def candidate_records(path: Path, *, names: tuple[str, ...] | None = None) -> Iterator[records.Cands]:
    """Inspect production candidate frames with the same typed worker decoder."""
    from trustmebro.artifacts import decode_nat_ext

    decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
    with closing(archives.candidate_frames(path, names)) as frames:
        for data in frames:
            yield archives.decode_candidates(data, decoder)


def fixture() -> r.Theorem:
    exprs = (
        r.Const("f", ()),
        r.Const("g", ()),
        r.Fvar(7),
        r.Fvar(8),
        r.App(1, (2, 3)),
        r.App(0, (2, 4, 2)),
        r.App(0, (2, 4, 2, 3)),
        r.NatLiteral(1 << 15000),
        r.Bvar(0),
        r.Forall(("a", "b"), 7, 8, r.BinderInfo.IMPLICIT),
        r.ExprMvar(7),
        r.App(0, (10,)),
        r.Const("unobserved", ()),
    )
    hyps = tuple(
        r.LocalConst(idx, ref, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT)
        for idx, ref in enumerate((4, 4, 9, 11, 7))
    )
    trns = tuple(
        r.Trn(r.Tactic("Lean.Parser.Tactic.exact", "exact h"), (idx, idx + 1), 1, r.ProofState(goal, hyps, (), ()))
        for idx, goal in enumerate((5, 6))
    )
    return r.Theorem("Fixture.first", "Fixture", (0, 20), exprs, trns)


def renumber(theorem: r.Theorem) -> r.Theorem:
    # A different valid post-order; arbitrary permutations are not source data.
    order = (12, 10, 8, 7, 3, 2, 1, 0, 11, 9, 4, 6, 5)
    refs = {old: new for new, old in enumerate(order)}
    exprs: list[r.Expr] = []
    for old in order:
        expr = theorem.exprs[old]
        match expr:
            case r.App(fn=fn, args=args):
                expr = r.App(refs[fn], tuple(refs[arg] for arg in args))
            case r.Forall(names=names, type=domain, body=body, binder_info=info):
                expr = r.Forall(names, refs[domain], refs[body], info)
        exprs.append(expr)
    trns = tuple(
        msgspec.structs.replace(
            trn,
            state=msgspec.structs.replace(
                trn.state,
                target=refs[trn.state.target],
                locals=tuple(msgspec.structs.replace(local, type=refs[local.type]) for local in trn.state.locals),
            ),
        )
        for trn in theorem.trns
    )
    return r.Theorem("Fixture.renumbered", theorem.module, theorem.src_span, tuple(exprs), trns)


def store(path: Path, theorems: tuple[r.Theorem, ...]) -> None:
    with closing(open_extraction_db(path)) as db:
        for theorem in theorems:
            db.execute(
                "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES(?,?,?,?,?,?)",
                (
                    theorem.name,
                    theorem.module,
                    len(theorem.exprs),
                    len(theorem.trns),
                    encode_exprs(theorem.exprs),
                    encode_trns(theorem.trns),
                ),
            )
        db.commit()


def operands(expr: r.Expr) -> tuple[int, ...]:
    # Independent fixture oracle, intentionally not the production graph helper.
    if isinstance(expr, r.App):
        return (expr.fn, *expr.args)
    if isinstance(expr, r.Forall):
        return expr.type, expr.body
    return ()


def reachable(exprs: tuple[r.Expr, ...], root: int) -> set[int]:
    visited: set[int] = set()
    pending = [root]
    while pending:
        ref = pending.pop()
        if ref not in visited:
            visited.add(ref)
            pending.extend(operands(exprs[ref]))
    return visited


def reference_fragment(exprs: tuple[r.Expr, ...], root: int, radius: int) -> tuple[records.Edges, tuple[int, ...]]:
    pending = deque([(root, 0)])
    nodes = [root]
    positions = {root: 0}
    edges: list[tuple[int, ...] | None] = []
    while pending:
        ref, depth = pending.popleft()
        children = operands(exprs[ref])
        if depth == radius and children:
            edges.append(None)
            continue
        local: list[int] = []
        for child in children:
            if child not in positions:
                positions[child] = len(nodes)
                nodes.append(child)
                pending.append((child, depth + 1))
            local.append(positions[child])
        edges.append(tuple(local))
    return tuple(edges), tuple(nodes)


class CandidateTests(unittest.TestCase):
    def test_deferred_database_validation_still_rejects_invalid_records_before_publication(self) -> None:
        # Failures: consolidating DB decoding accidentally drops reference/span
        # validation or publishes a partial archive. Exercises the complete
        # storage -> worker -> publication boundary, not just an unchecked helper.
        source = fixture()
        bad_refs = msgspec.structs.replace(source, exprs=(r.App(0, (0,)), *source.exprs[1:]))
        bad_span = msgspec.structs.replace(
            source, trns=(msgspec.structs.replace(source.trns[0], src_span=(5, 2)), *source.trns[1:])
        )
        for bad in (bad_refs, bad_span):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                db, output = Path(tmp) / "bad.db", Path(tmp) / "candidates.zst"
                store(db, (bad,))
                with self.assertRaises(r.SchemaError):
                    scheduler.scan_cands(db, output, workers=1)
                self.assertFalse(output.exists())

    def test_database_to_archive_preserves_fragments_positions_and_occurrences(self) -> None:
        # Failures: discovery-tree edges substituted for DAG edges; arity/order
        # lost; ID-dependent identities; wildcard and leaf conflated; repeated
        # hypotheses or shared paths misweighted; literals/binders/vars changed;
        # unused nodes emitted; repeated depth/state searches. Establishes exact
        # fixture behavior and shared preparation, not wildcard matching or ML.
        first = fixture()
        second = renumber(first)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "source.db"
            output = root / "candidates.msgpack.zst"
            store(db, (first, second))
            original = db.read_bytes()
            with patch.object(graph_ops, "shortest_distance", wraps=graph_ops.shortest_distance) as searches:
                counts = scheduler.scan_cands(db, output, depths=(3, 1, 2, 2))
            observed = tuple(candidate_records(output))
            self.assertEqual(counts.theorems, 2)
            # Failures: header/framing excluded, worker byte counts confused
            # with final file bytes, or a second encoding changes the reported size.
            self.assertEqual(counts.stored_bytes, output.stat().st_size)
            self.assertEqual(counts.uncompressed_stream_bytes, len(zstd.decompress(output.read_bytes())))
            self.assertEqual(archives.read_selection(output).depths, (1, 2, 3))
            self.assertEqual(db.read_bytes(), original)
            self.assertEqual(searches.call_count, 2 * sum(bool(operands(expr)) for expr in first.exprs))
            self.assertTrue(all(call.kwargs["max_dist"] == 3 for call in searches.call_args_list))
            identities = []
            for source, row in zip((first, second), observed, strict=True):
                self.assertEqual(row.name, source.name)
                goals: Counter[int] = Counter()
                hyps: Counter[int] = Counter()
                for state, trn in zip(row.states, source.trns, strict=True):
                    self.assertEqual(state.tactic, trn.tactic)
                    self.assertEqual(state.hyps, tuple(local.type for local in trn.state.locals))
                    goals.update(reachable(source.exprs, state.goal))
                    for hyp in state.hyps:
                        hyps.update(reachable(source.exprs, hyp))
                for node in row.nodes:
                    self.assertEqual(
                        (node.expr, node.goal_count, node.hyp_count),
                        (source.exprs[node.ref], goals[node.ref], hyps[node.ref]),
                    )
                self.assertEqual({node.ref for node in row.nodes}, goals.keys() | hyps.keys())
                for closure in row.roots:
                    self.assertEqual(set(closure.anchors), reachable(source.exprs, closure.ref))
                for occurrence in row.occs:
                    shape = row.shapes[occurrence.shape]
                    edges, refs = reference_fragment(source.exprs, occurrence.nodes[0], occurrence.depth)
                    self.assertEqual((shape.edges, occurrence.nodes), (edges, refs))
                    self.assertEqual(shape.ident, hashlib.sha256(msgspec.msgpack.encode(edges)).digest())
                identities.append({shape.ident for shape in row.shapes})
            self.assertEqual(identities[0], identities[1])
            first_row = observed[0]
            shaped = {(occ.nodes[0], occ.depth): first_row.shapes[occ.shape] for occ in first_row.occs}
            self.assertEqual(shaped[5, 1].edges, ((1, 2, 3, 2), (), (), None))
            self.assertNotEqual(shaped[5, 1].ident, shaped[6, 1].ident)
            self.assertEqual(shaped[7, 1], shaped[7, 3])
            self.assertEqual(len(first_row.occs), 3 * len(first_row.nodes))
            self.assertEqual(next(node.expr for node in first_row.nodes if node.ref == 7), r.NatLiteral(1 << 15000))

    def test_ready_results_replenish_the_bounded_queue_before_consumption(self) -> None:
        # Failures: a blocked first job hides ready results; refill waits for the
        # consumer; the initial queue is unbounded; rows are lost or duplicated.
        # This scheduling boundary needs controlled completion, which natural
        # DB-to-archive fixtures cannot guarantee. It does not measure throughput.
        release_first, third_running = Event(), Event()
        submitted: list[str] = []

        def rows() -> Iterator[scheduler.SrcRow]:
            for idx in range(6):
                name = str(idx)
                submitted.append(name)
                yield name, "", None, None, 0, 0, b"", b""

        def worker(row: scheduler.SrcRow) -> tuple[bytes, scheduler.CandCounts]:
            if row[0] == "0" and not release_first.wait(10):
                raise TimeoutError("the ready job was blocked by submission order")
            if row[0] == "2":
                third_running.set()
            return row[0].encode(), scheduler.CandCounts(theorems=1)

        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(scheduler, "_cand_worker", side_effect=worker):
            results = scheduler.ready_results(pool, rows(), workers=1, task=scheduler._cand_worker)
            try:
                self.assertEqual(next(results)[0], b"1")
                self.assertTrue(third_running.wait(5), "replacement job was not queued before yielding")
                self.assertEqual(submitted, ["0", "1", "2"])
                release_first.set()
                self.assertCountEqual([data for data, _ in results], [b"0", b"2", b"3", b"4", b"5"])
            finally:
                release_first.set()
                results.close()

        # Failures: multiple completed jobs free slots but only one is refilled
        # before a slow consumer; refilling exceeds the outstanding-job bound;
        # completion batches lose/duplicate results. Force an entire batch to
        # finish together, a timing condition end-to-end fixtures cannot ensure.
        submitted_indices: list[int] = []
        batch_sizes: list[int] = []

        def indices() -> Iterator[int]:
            for idx in range(12):
                submitted_indices.append(idx)
                yield idx

        def complete_batch(futures: set[Future[int]], *, return_when: str) -> tuple[set[Future[int]], set[Future[int]]]:
            batch_sizes.append(len(futures))
            return wait(futures)  # ALL_COMPLETED: deliberately exercise batching.

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(runtime, "wait", side_effect=complete_batch),
            closing(scheduler.ready_results(pool, indices(), workers=2, task=abs)) as results,
        ):
            first = next(results)
            self.assertEqual(submitted_indices, list(range(8)), "all four slots must refill before yielding")
            self.assertCountEqual([first, *results], range(12))
            self.assertTrue(batch_sizes)
            self.assertLessEqual(max(batch_sizes), 4)

    def test_failure_preserves_source_and_previous_archive(self) -> None:
        # Failures: existing output overwritten without consent; source used as
        # output; missing selection silently accepted; invalid/cyclic references
        # published; failed producer leaves partial data; truncated frame accepted.
        # Establishes publication/failure behavior, not concurrent-source safety.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "source.db"
            output = root / "candidates.zst"
            store(db, (fixture(),))
            scheduler.scan_cands(db, output)
            original_db, original_output = db.read_bytes(), output.read_bytes()
            with self.assertRaises(FileExistsError):
                scheduler.scan_cands(db, output)
            with self.assertRaises(ValueError):
                scheduler.scan_cands(db, db, replace=True)
            with self.assertRaises(ValueError):
                scheduler.scan_cands(db, output, theorems=("Fixture.first", "missing"), replace=True)
            self.assertEqual(db.read_bytes(), original_db)
            self.assertEqual(output.read_bytes(), original_output)
            cyclic = msgspec.structs.replace(
                fixture(),
                name="Fixture.bad",
                exprs=(r.App(0, (0,)),),
                trns=(msgspec.structs.replace(fixture().trns[0], state=r.ProofState(0, (), (), ())),),
            )
            bad_db = root / "cyclic.db"
            store(bad_db, (fixture(), cyclic))
            with self.assertRaisesRegex(ValueError, "post-order"):
                scheduler.scan_cands(bad_db, output, workers=2, replace=True)
            self.assertEqual(output.read_bytes(), original_output)
            self.assertEqual(list(root.glob("*.part")), [])
            broken = root / "broken.zst"
            broken.write_bytes(zstd.compress(b"\x00\x01"))
            with self.assertRaisesRegex(ValueError, "truncated"):
                tuple(candidate_records(broken))


if __name__ == "__main__":
    unittest.main()
