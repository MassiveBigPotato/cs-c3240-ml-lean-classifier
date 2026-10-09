"""End-to-end identity, multiplicity, parallelism and render-cache checks."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from analysis_fixture import analyze

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.visualization.archives import read_stats, row_vals
from trustmebro.visualization.products import AnalysisPaths


class ExpressionFrequencyTests(unittest.TestCase):
    def test_database_to_exact_counts_and_freq_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir = Path(tmp)
            src = dir / "source.db"

            def exprs(x, y, name="f"):
                return (r.Fvar(x), r.Fvar(y), r.Const(name, ()), r.App(2, (0,)), r.App(3, (0,)), r.App(3, (1,)))

            def trn(root, local=False):
                locals_ = (
                    (r.LocalConst(9, root, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT),) if local else ()
                )
                return r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(root, locals_, (), ()))

            deep = [r.Const("deep", ())]
            for _ in range(1100):
                deep.append(r.App(len(deep) - 1, (len(deep) - 1,)))
            rows = [
                ("original", exprs(10, 20), (trn(4, True), trn(5))),
                ("renamed", exprs(100, 200), (trn(4),)),
                ("different_constant", exprs(10, 20, "g"), (trn(4),)),
                ("deep", tuple(deep), (trn(1100),)),
                (
                    "universes",
                    (
                        r.Const("u", (r.LvlMvar(8), r.LvlMvar(8))),
                        r.Const("u", (r.LvlMvar(90), r.LvlMvar(90))),
                        r.Const("u", (r.LvlMvar(90), r.LvlMvar(91))),
                        r.ExprMvar(10),
                        r.Fvar(10),
                    ),
                    tuple(trn(root) for root in range(5)),
                ),
            ]
            with closing(open_extraction_db(src)) as database:
                database.executemany(
                    """INSERT INTO theorems
                    (name,module,expr_count,trn_count,exprs,trns)
                    VALUES (?, 'Test', ?, ?, ?, ?)""",
                    [
                        (name, len(exprs), len(steps), encode_exprs(exprs), encode_trns(steps))
                        for name, exprs, steps in rows
                    ],
                )
            original = src.read_bytes()
            env = dict(os.environ, MPLCONFIGDIR=str(dir / "matplotlib"))
            command = [
                str(Path(sys.executable).parent / "graphs"),
                "--db",
                str(src),
                "--graphs",
                "complexity",
                "reuse",
                "frequencies",
            ]
            snapshots = []
            for workers in (1, 2):
                output = dir / str(workers)
                invocation = [*command, "--output", str(output), "--stats", str(output), "--workers", str(workers)]
                if workers == 1:
                    subprocess.run(invocation, check=True, capture_output=True, env=env, timeout=120)
                else:
                    analyze(src, output, analyses=("metrics",), workers=workers)
                cache = AnalysisPaths(output).analysis("metrics")
                rows = sorted(
                    row_vals(row) for record in read_stats(cache) if record[0] == "freqs" for row in record[1]
                )
                snapshots.append(rows)
                totals = [sum(int(row[col]) for row in rows) for col in (3, 4, 5)]
                self.assertEqual(totals, [11, 1127, 2**1101 + 29])
                # Renaming preserves sharing; f x x and g x x have the same
                # sizes but different identities; f x y remains distinct.
                self.assertEqual(sorted(row[3] for row in rows if row[1:3] == (4, 5)), [1, 3])
                self.assertEqual([row[3] for row in rows if row[1:3] == (5, 5)], [1])
                self.assertEqual(
                    [row[3:6] for row in rows if row[1:3] == (1, 1) and row[5] == (2**1100)], [(0, 1, 2**1100)]
                )
                self.assertEqual([(row[2], row[3]) for row in rows if row[1] == 1101], [(2**1101 - 1, 1)])
                self.assertEqual(
                    sorted(int(row[3]) for row in rows if row[1:3] == (1, 1) and row[3] != 0), [1, 1, 1, 2]
                )
                self.assertFalse(list(output.glob("*.sqlite")))
                self.assertFalse(list(output.glob("*.part")))
                if workers == 1:
                    for pop in ("top_level", "distinct_subexprs"):
                        for statistic in ("total",):
                            self.assertTrue(
                                (output / f"freq-{pop}-{statistic}.png").read_bytes().startswith(b"\x89PNG")
                            )
                    for name in (
                        "subexpression-repetition.png",
                        "vocabulary-coverage.png",
                        "reuse-breadth.png",
                        "complexity-state.png",
                    ):
                        self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG"))
                    coverage = json.loads((output / "vocabulary-coverage.json").read_text())
                    self.assertEqual(sorted(row[6] for row in rows if row[1:3] == (4, 5)), [1, 2])
                    self.assertEqual(coverage["top_level"]["retained"][0], 1)
                    self.assertAlmostEqual(coverage["top_level"]["coverage"][0], 3 / 11)
                    self.assertAlmostEqual(coverage["expanded_subexprs"]["coverage"][0], 0.5)
                    for curve in coverage.values():
                        self.assertAlmostEqual(curve["coverage"][-1], 1)
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(src.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
