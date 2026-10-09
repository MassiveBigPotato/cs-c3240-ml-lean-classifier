"""Real Lean exports exercise ordered slots, partial heads and trailing args."""

import io
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from trustmebro.extraction.records import App, Const, Fvar, iter_theorems
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.graph import ReachCache, build_graph, graph_stats
from trustmebro.visualization.measurements import ViewMode
from trustmebro.visualization.views import (
    Kind,
    OpName,
    Role,
    app_spine,
    expr_views,
    match_rule,
    measure_view,
    resolve_root,
    view_arrs,
)


class PlumbingTests(unittest.TestCase):
    def test_exported_applications_keep_operand_layouts_and_stored_data(self) -> None:
        # Failures: Lean normalization leaves registered casts/operators raw,
        # erases the wrong coercion operand, or changes ordered trailing args;
        # Python projections mutate stored exports or change root populations.
        # Checks the normalized export, not reconstruction of source Exprs.
        project = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            ["lake", "exe", "trustmebro-extract-state", "src/trustmebro/extraction/Fixture/Plumbing.lean"],
            cwd=project,
            capture_output=True,
            check=False,
            timeout=300,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        theorems = {theorem.name.rsplit(".", 1)[-1]: theorem for theorem in iter_theorems(io.BytesIO(process.stdout))}
        seen: set[str] = set()
        partial = False
        trailing = False
        instance = False
        membership_layout = False
        normalized_operator = False
        for theorem in theorems.values():
            exprs = theorem.exprs
            original = tuple(exprs)
            instances, compact, erased = expr_views(build_graph(exprs))[1:]
            erased_stats, erased_cache = graph_stats(erased.graph), ReachCache(erased.graph)
            arrays = view_arrs(erased)
            for root, expr in enumerate(exprs):
                if not isinstance(expr, App):
                    continue
                head, args = app_spine(exprs, root)
                head_expr = exprs[head]
                match = match_rule(exprs, root)
                if isinstance(head_expr, Const) and head_expr.name == "Add.add":
                    self.assertEqual(len(args), 2)
                    normalized_operator = True
                if isinstance(head_expr, Const) and head_expr.name == "HAdd.hAdd" and len(args) < 6:
                    self.assertIsNone(match)
                    partial = True
                if match is None:
                    continue
                seen.add(match.rule.head)
                self.assertEqual((*match.args, *match.tail), args)
                if match.rule.kind in (Kind.FN_COE, Kind.VAL_COE, Kind.TYPE_COE, Kind.SET_COE, Kind.PROJ):
                    marker = compact.markers[root]
                    self.assertEqual(marker.kind, match.rule.kind)
                    self.assertEqual(marker.fixed, match.rule.fixed)
                    self.assertEqual(
                        {operand.name: operand.ref for operand in marker.slots},
                        {
                            slot.name: ref
                            for slot, ref in zip(match.rule.slots, match.args, strict=True)
                            if slot.role != Role.INST
                        },
                    )
                    val = match.arg(
                        OpName.OBJ if match.rule.kind in (Kind.FN_COE, Kind.SET_COE, Kind.PROJ) else OpName.VAL
                    )
                    if match.tail:
                        resolved = resolve_root(erased, val)
                        prefix = (resolved,)
                        val_marker = erased.markers.get(resolved)
                        if isinstance(exprs[resolved], App) and (
                            val_marker is None or val_marker.kind == Kind.OPERATOR
                        ):
                            prefix = erased.graph.edges[resolved]
                        self.assertEqual(
                            erased.graph.edges[root], (*prefix, *(resolve_root(erased, ref) for ref in match.tail))
                        )
                    else:
                        self.assertEqual(resolve_root(erased, root), resolve_root(erased, val))
                        self.assertEqual(
                            measure_view(erased, erased_stats, erased_cache, arrays, [root]),
                            measure_view(erased, erased_stats, erased_cache, arrays, [val]),
                        )
                    self.assertNotIn(root, instances.markers)
                if match.rule.head == "Membership.mem":
                    state = theorem.trns[0].state
                    locals_by_id = {local.id: local for local in state.locals}
                    vals = {name: exprs[match.arg(name)] for name in (OpName.CONTAINER, OpName.ELEM)}
                    # Closed expressions stored alongside states have bvars;
                    # check the open proof-state forms against their locals.
                    if all(isinstance(val, Fvar) for val in vals.values()):
                        for val_name, val in vals.items():
                            assert isinstance(val, Fvar)
                            self.assertEqual(locals_by_id[val.id].type, match.arg(OpName(f"{val_name}_type")))
                        membership_layout = True
                if match.rule.head == "HAdd.hAdd" and match.tail:
                    self.assertEqual(len(match.tail), 1)
                    trailing = True
                for slot, ref in zip(match.rule.slots, match.args, strict=True):
                    if slot.role == Role.INST:
                        scoped = match_rule(exprs, ref, inst_ctxt=True)
                        if scoped is not None:
                            self.assertIsNone(match_rule(exprs, ref))
                            instance = True
            self.assertEqual(exprs, original)
        self.assertTrue({"Membership.mem", "HAdd.hAdd", "OfNat.ofNat", "Subtype.val", "DFunLike.coe"} <= seen)
        self.assertTrue(normalized_operator)
        for name in ("cast", "genericCast"):
            theorem = theorems[name]
            goal = theorem.exprs[theorem.trns[0].state.target]
            self.assertIsInstance(goal, App)
            assert isinstance(goal, App)
            head = theorem.exprs[goal.fn]
            self.assertIsInstance(head, Const)
            assert isinstance(head, Const)
            self.assertEqual(head.name, "Eq")
            self.assertEqual(len(goal.args), 3)
            lhs, rhs = (theorem.exprs[ref] for ref in goal.args[-2:])
            self.assertIsInstance(lhs, Fvar)
            self.assertEqual(lhs, rhs)
            self.assertFalse(
                any(isinstance(expr, Const) and expr.name in ("Nat.cast", "Coe.coe") for expr in theorem.exprs)
            )
        theorem = theorems["typeCoercion"]
        locals = theorem.trns[0].state.locals
        nat_ids = {local.id for local in locals if theorem.exprs[local.type] == Const("Nat", ())}
        self.assertTrue(
            any(isinstance(expr := theorem.exprs[local.type], Fvar) and expr.id in nat_ids for local in locals)
        )
        self.assertTrue(partial)
        self.assertTrue(trailing)
        self.assertTrue(instance)
        self.assertTrue(membership_layout)
        # Export → durable DB → exact corpus scan → numeric statistics → plots.
        # Source bounds are test fixtures, not sampling inside the scanner.
        from analysis_fixture import analyze, read_pattern_cols

        from trustmebro.visualization import stral_render
        from trustmebro.visualization.measurements import PATTERN_MODES
        from trustmebro.visualization.products import AnalysisPaths

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            db_path = out / "source.db"
            with closing(open_extraction_db(db_path)) as db:
                for theorem in theorems.values():
                    db.execute(
                        "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES (?,?,?,?,?,?)",
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
            original = db_path.read_bytes()
            analyze(db_path, out, analyses=("patterns",), workers=1, depths=(1, 2))
            expected = sum(1 + len(trn.state.locals) for theorem in theorems.values() for trn in theorem.trns)
            for mode in PATTERN_MODES:
                for depth in (1, 2):
                    heads, array = read_pattern_cols(AnalysisPaths(out).patterns(mode, depth))
                    for flavour in (0, 1):
                        rows = array[(array[:, 0] == flavour) & (array[:, 1] == 0)]
                        self.assertEqual(int(rows[:, 2].sum()), expected)
                        self.assertGreater(int(rows[:, 4].sum()), 0)
                        self.assertLess(int(rows[:, 4].sum()), expected)
                    self.assertIn("DFunLike.coe", heads)
            # Compare semantic group identities, not worker-dependent head IDs.
            baseline_heads, baseline = read_pattern_cols(AnalysisPaths(out).patterns(ViewMode.ORIGINAL, 1))
            baseline_weights = {
                (baseline_heads[int(group[0, 1])], flavour): tuple(int(x) for x in group[:, 2:].sum(axis=0))
                for flavour in (0, 1)
                for head in range(len(baseline_heads))
                if len(group := baseline[(baseline[:, 0] == flavour) & (baseline[:, 1] == head)])
            }
            for mode in PATTERN_MODES[1:]:
                heads, array = read_pattern_cols(AnalysisPaths(out).patterns(mode, 1))
                weights = {
                    (heads[int(group[0, 1])], flavour): tuple(int(x) for x in group[:, 2:].sum(axis=0))
                    for flavour in (0, 1)
                    for head in range(len(heads))
                    if len(group := array[(array[:, 0] == flavour) & (array[:, 1] == head)])
                }
                self.assertEqual(weights, baseline_weights)
            stral_render.render_patterns(out, out)
            self.assertTrue(
                (out / "local-pattern-topology-top-affected-depth1.png").read_bytes().startswith(b"\x89PNG")
            )
            self.assertTrue((out / "local-pattern-heads-0-depth2.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((out / "local-pattern-heads.csv.zst").is_file())
            self.assertEqual(db_path.read_bytes(), original)
