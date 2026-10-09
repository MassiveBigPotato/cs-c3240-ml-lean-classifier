"""Analysis product discovery, immutable generations and single-commit publication."""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Callable, Generator, Iterable, Mapping
from compression import zstd
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Literal

import msgspec

from trustmebro.visualization.archives import read_summary, write_summary

from .measurements import PATTERN_MODES, VIEW_MODES, ViewMode

type Analysis = Literal["metrics", "topology", "patterns"]
MANIFEST = "analysis-manifest.msgpack.zst"


class SrcStamp(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    size: int
    modified_ns: int
    limit: int | None

    @classmethod
    def from_path(cls, path: Path, limit: int | None) -> SrcStamp:
        stat = path.stat()
        return cls(str(path.resolve()), stat.st_size, stat.st_mtime_ns, limit)


@dataclass(frozen=True)
class AnalysisCfg:
    atlas_min_nodes: int
    depths: tuple[int, ...]
    embedding: bool = False


class Gen(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    dir: str
    src: SrcStamp
    cfg: AnalysisCfg
    artifacts: tuple[str, ...]
    sizes: tuple[int, ...]
    format: Literal["columns-v1"]


@dataclass(frozen=True)
class AnalysisPaths:
    """One naming contract shared by producers, renderers, and CLI checks."""

    root: Path

    @cached_property
    def manifest(self) -> dict[Analysis, Gen]:
        path = self.root / MANIFEST
        return read_summary(path, dict[Analysis, Gen]) if path.is_file() else {}

    def dir(self, kind: Analysis) -> Path:
        if (gen := self.manifest.get(kind)) is None:
            return self.root
        if not gen.dir.startswith("run-") or Path(gen.dir).name != gen.dir:
            raise ValueError("invalid analysis-generation directory")
        return self.root / ".generations" / gen.dir

    def artifact(self, kind: Analysis, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError("invalid analysis-artifact name")
        return self.dir(kind) / name

    def require(self, kinds: Iterable[Analysis], artifacts: Mapping[Analysis, Iterable[str]]) -> None:
        """Validate a pinned manifest and the selected consumers' dependencies."""
        srcs: set[SrcStamp] = set()
        for kind in kinds:
            gen = self.manifest.get(kind)
            if gen is None:
                raise ValueError(f"no published {kind} analysis; run graphs --replace")
            srcs.add(gen.src)
            if len(gen.sizes) != len(gen.artifacts):
                raise ValueError("invalid statistics manifest; regenerate with --replace")
            sizes = dict(zip(gen.artifacts, gen.sizes, strict=True))
            for name in artifacts[kind]:
                if name not in gen.artifacts or not self.artifact(kind, name).is_file():
                    raise ValueError(f"incomplete {kind} analysis: missing {name}; rerun graphs --replace")
                if self.artifact(kind, name).stat().st_size != sizes[name]:
                    raise ValueError(f"damaged {kind} statistics: {name}; regenerate with --replace")
        if len(srcs) > 1:
            raise ValueError("selected analyses describe different source selections; regenerate them together")

    def analysis(self, kind: Analysis) -> Path:
        names = {
            "metrics": "metrics.msgpack.zst",
            "topology": "topology-counts.msgpack.zst",
            "patterns": "local-patterns.msgpack.zst",
        }
        return self.artifact(kind, names[kind])

    def topo(self, mode: ViewMode = ViewMode.ORIGINAL) -> Path:
        suffix = "" if mode == "original" else f"-{mode}"
        return self.artifact("topology", f"topology-counts{suffix}.msgpack.zst")

    def patterns(self, mode: ViewMode, depth: int) -> Path:
        return self.artifact("patterns", f"local-patterns-{mode}-{depth}.msgpack.zst")

    def stats(self, kind: Literal["topology", "comparison", "head", "pattern"]) -> Path:
        return self.artifact("patterns" if kind == "pattern" else "topology", f"{kind}-statistics.msgpack.zst")

    def embedding(self, mode: str) -> Path:
        if mode not in ("size-aware", "shape-only"):
            raise ValueError(f"unknown embedding mode: {mode}")
        suffix = "" if mode == "size-aware" else "-shape-only"
        return self.artifact("metrics", f"structural-embedding{suffix}.npz")

    @property
    def embedding_shared(self) -> Path:
        return self.artifact("metrics", "structural-observations.npz")

    @property
    def embedding_info(self) -> Path:
        return self.artifact("metrics", "embedding.msgpack.zst")

    @property
    def comparisons(self) -> Path:
        return self.artifact("topology", "topology-comparisons.msgpack.zst")

    @property
    def examples(self) -> Path:
        return self.artifact("topology", "topology-examples.msgpack.zst")

    @property
    def heads(self) -> Path:
        return self.artifact("topology", "function-heads.msgpack.zst")


type RenderStage = Literal[
    "metrics", "topology", "comparisons", "heads", "topology-atlas", "local-patterns", "candidates", "features"
]


@dataclass(frozen=True)
class GraphFamily:
    analysis: Analysis | None
    descr: str
    artifacts: Callable[[AnalysisPaths], tuple[Path, ...]] = lambda paths: (paths.analysis("metrics"),)
    stage: RenderStage = "metrics"


# One registry governs CLI choices, dependencies, and render dispatch.
GRAPH_FAMILIES: Mapping[str, GraphFamily] = MappingProxyType(
    {
        "complexity": GraphFamily("metrics", "State size, expansion, and context density"),
        "constructors": GraphFamily("metrics", "Expression constructor composition"),
        "context": GraphFamily("metrics", "Context composition and concentration"),
        "reuse": GraphFamily("metrics", "Global reuse, novelty, and vocabulary coverage"),
        "frequencies": GraphFamily("metrics", "Expression occurrence distributions"),
        "embedding": GraphFamily(
            "metrics",
            "Structural features and UMAP embeddings",
            lambda paths: (paths.analysis("metrics"), paths.embedding_info, paths.embedding_shared),
        ),
        "expr-atlas": GraphFamily("metrics", "Expression DAGs, adjacency, and layer profiles"),
        "topology": GraphFamily(
            "topology",
            "Complete DAG topology frequencies and coverage",
            lambda paths: (paths.stats("topology"), *(paths.topo(mode) for mode in VIEW_MODES)),
            stage="topology",
        ),
        "comparisons": GraphFamily(
            "topology",
            "Coercion and instance transformations versus the exported baseline",
            lambda paths: (paths.comparisons, paths.stats("comparison")),
            stage="comparisons",
        ),
        "heads": GraphFamily(
            "topology", "Application-head frequencies", lambda paths: (paths.heads, paths.stats("head")), stage="heads"
        ),
        "topology-atlas": GraphFamily(
            "topology",
            "Common topologies and paired graph examples",
            lambda paths: (paths.topo(), paths.examples),
            stage="topology-atlas",
        ),
        "local-patterns": GraphFamily(
            "patterns",
            "Local patterns and head-conditioned coverage",
            lambda paths: (paths.analysis("patterns"), paths.stats("pattern")),
            stage="local-patterns",
        ),
        "candidates": GraphFamily(
            None, "Selected vocabulary shapes and actual feature allocation", lambda paths: (), stage="candidates"
        ),
        "features": GraphFamily(
            None,
            "Actual feature coverage, activation and bounded pattern co-occurrence",
            lambda paths: (),
            stage="features",
        ),
    }
)


def _products(paths: AnalysisPaths, kind: Analysis, cfg: AnalysisCfg) -> tuple[Path, ...]:
    """The graph registry defines product requirements; production adds full populations."""
    required = dict.fromkeys(
        path
        for name, family in GRAPH_FAMILIES.items()
        if family.analysis == kind and (name != "embedding" or cfg.embedding)
        for path in family.artifacts(paths)
    )
    if kind == "metrics" and cfg.embedding:
        required.update(
            dict.fromkeys(paths.embedding(mode) for mode in read_summary(paths.embedding_info, tuple[str, ...]))
        )
    if kind == "patterns":
        required.update(dict.fromkeys(paths.patterns(mode, depth) for mode in PATTERN_MODES for depth in cfg.depths))
    return tuple(required)


@contextmanager
def analysis_set(
    root: Path,
    kinds: tuple[Analysis, ...],
    src: Path,
    *,
    limit: int | None = None,
    cfg: AnalysisCfg,
    replace: bool = False,
) -> Generator[AnalysisPaths]:
    """One manifest replacement publishes the entire request, including embeddings.

    Readers pin immutable generations. Failed producers leave the old manifest
    intact; unselected analyses retain their previous generation. The directory
    lock prevents concurrent publishers from losing each other's manifest edits.
    """
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    gens = root / ".generations"
    published: Path | None = None
    committed = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prev = AnalysisPaths(root)
        try:
            previous = prev.manifest
        except OSError, ValueError, zstd.ZstdError:
            if not replace:
                raise ValueError("obsolete/damaged statistics manifest; regenerate with --replace") from None
            previous = {}
        kinds = msgspec.convert(kinds, type=tuple[Analysis, ...])
        if not kinds:
            raise ValueError("at least one analysis is required")
        for kind in kinds:
            if not replace and (
                kind in previous
                or (root / ("metrics.msgpack.zst" if kind == "metrics" else kind + ".msgpack.zst")).exists()
            ):
                raise FileExistsError(f"{kind} measurements already exist; use --replace")
        stamp = SrcStamp.from_path(src, limit)
        gens.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".pending-", dir=gens) as tmp:
            pending = AnalysisPaths(Path(tmp))
            yield pending
            if SrcStamp.from_path(src, limit) != stamp:
                raise ValueError("source database changed during analysis")
            manifest = previous.copy()
            dir_name = "run-" + pending.root.name.removeprefix(".pending-")
            for kind in kinds:
                products = _products(pending, kind, cfg)
                if any(not path.is_file() for path in products):
                    raise ValueError(f"cannot publish incomplete {kind} analysis")
                manifest[kind] = Gen(
                    dir_name,
                    stamp,
                    cfg,
                    tuple(path.name for path in products),
                    tuple(path.stat().st_size for path in products),
                    "columns-v1",
                )
            published = gens / dir_name
            pending.root.rename(published)
            write_summary(root / MANIFEST, manifest)
            committed = True
    finally:
        if published is not None and not committed:
            # A signal can arrive after the atomic swap but before `committed`
            # is assigned. Never remove a generation already referenced by it.
            try:
                active = AnalysisPaths(root).manifest.values()
                refd = any(gen.dir == published.name for gen in active)
            except OSError, ValueError, msgspec.DecodeError, msgspec.ValidationError:
                refd = True  # uncertain publication: retain recoverable data
            if not refd:
                shutil.rmtree(published)
        if gens.exists():
            try:
                gens.rmdir()  # only an empty, newly created container
            except OSError:
                pass
        os.close(descriptor)
