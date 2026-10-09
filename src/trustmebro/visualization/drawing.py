"""Shared raster drawing, logarithmic axis labels, and comparable density panels."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import datashader as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datashader.compiler import compile_components
from datashader.glyphs import Point
from datashader.utils import dshape_from_pandas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, Locator, MultipleLocator
from scipy.ndimage import maximum_filter as max_filter

if TYPE_CHECKING:
    from graph_tool import Graph, GraphView, PropertyMap
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis as MplAxis
    from matplotlib.figure import Figure
    from xarray import DataArray


# Plot specifications and prepared raster records.


type PointCols = dict[str, np.ndarray]
type PointBatch = pd.DataFrame | Mapping[str, np.ndarray]
type FrameSrc = pd.DataFrame | Callable[[], Iterator[PointBatch]]

# Reader batches still bound individual theorems. Rendering combines them before
# pandas/native kernels; this separately bounds retained plot columns, not RSS.
POINT_BATCH_SIZE = 10_000
POINT_MEMORY_BUDGET = 1 << 30


@dataclass(frozen=True)
class PreparedPoints:
    chunks: list[PointCols] | None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None
    nbytes: int


@dataclass(frozen=True)
class GraphStyle:
    fill: str | PropertyMap = "#57bde9"
    size: float | PropertyMap = 0.01
    edges: str | PropertyMap = "#a8bdc9"
    edge_width: float = 0.004
    marker_size: float | None = None


@dataclass(frozen=True)
class Axis:
    """A prepared coordinate column and its explicit actual-value display projection."""

    field: str
    label: str
    proj: Literal["linear", "log10"] = "linear"


@dataclass(frozen=True)
class DensityPlot:
    filename: str
    title: str
    x: Axis
    y: Axis
    caption: str
    diag: bool = False


@dataclass(frozen=True)
class DensityCfg:
    median: str | None = None
    weight: str | None = None
    colour: str | None = None
    label: str | None = None
    # Colour fields are already projected; this controls display, not their data.
    colour_proj: Literal["linear", "log10"] = "log10"
    colour_range: tuple[float, float] | None = None
    reduce_color: Literal["mean", "max"] = "max"

    def __post_init__(self) -> None:
        if sum(field is not None for field in (self.median, self.weight, self.colour)) > 1:
            raise ValueError("choose one density reducer: median, weight, or colour")
        if self.colour_range is not None and self.colour_range[0] >= self.colour_range[1]:
            raise ValueError("colour range must have increasing bounds")


class DensityDrawer(Protocol):
    # DensityCfg is frozen with immutable fields, so sharing a default is safe.
    def __call__(self, dir: Path, frame: FrameSrc, plot: DensityPlot, cfg: DensityCfg = DensityCfg()) -> None: ...  # noqa: B008


@dataclass(frozen=True)
class Raster:
    canvas: ds.Canvas
    vals: np.ndarray
    label: str


@dataclass(frozen=True)
class PointRaster:
    """Bound native callbacks owning one reduction's buffers, not its input batches."""

    append: Callable[[pd.DataFrame], None]
    finish: Callable[[], DataArray]


@dataclass(frozen=True)
class PanelCfg:
    xlabel: str
    ylabel: str
    diag: bool = False
    eq_ranges: bool = False
    y_proj: Literal["linear", "log10"] = "log10"
    caption: str = ""
    x_proj: Literal["linear", "log10"] = "log10"


# Style, axes, and coordinate projection.


def plot_style[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Figures own their appearance without changing the caller's rc settings."""

    @wraps(fn)
    def draw(*args: P.args, **kwargs: P.kwargs) -> R:
        with plt.style.context("dark_background"), mpl.rc_context({"image.cmap": "turbo"}):
            return fn(*args, **kwargs)

    return draw


def actual_val_ticks(axis: MplAxis) -> None:
    """Label log10 coordinates without exponentiating enormous tree sizes."""

    def label(exp: float, _pos: int) -> str:
        if 0 <= exp <= 6 and exp.is_integer():
            return f"{10**exp:,.0f}"
        if -3 <= exp <= 6:
            return f"{10**exp:,.4g}"
        return f"1e{exp:g}"

    class MinorLogTicks(Locator):
        def __call__(self) -> list[float]:
            if self.axis is None:
                return []
            lo, hi = sorted(self.axis.get_view_interval())
            if hi - lo > 25:
                return []
            decades = np.arange(math.floor(lo), math.ceil(hi))
            ticks = (decades[:, None] + np.log10(np.arange(2, 10))).ravel()
            return ticks[(ticks >= lo) & (ticks <= hi)].tolist()

    lo, hi = axis.get_view_interval()
    axis.set_major_locator(MultipleLocator(max(1, math.ceil(abs(hi - lo) / 9))))
    axis.set_minor_locator(MinorLogTicks())
    axis.set_major_formatter(FuncFormatter(label))


def format_axis(axis: MplAxis, proj: Literal["linear", "log10"]) -> None:
    if proj == "log10":
        actual_val_ticks(axis)


def count_ticks(axis: MplAxis) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10, numticks=7))
    axis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    axis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}" if val < 1e7 else f"{val:.0e}"))
    axis.set_minor_formatter(NullFormatter())


def frame_batches(src: FrameSrc) -> Iterator[PointBatch]:
    """Replayable columns or frames; pandas is only required at the raster boundary."""
    if isinstance(src, pd.DataFrame):
        yield src
    else:
        yield from src()


def log_counts(vals: np.ndarray) -> np.ndarray:
    """Rendering projection: native arrays use NumPy; huge integers never cast to float first."""
    if vals.dtype == object:
        return np.fromiter((math.log10(val) if val > 0 else np.nan for val in vals), dtype=np.float64, count=len(vals))
    return np.log10(vals, out=np.full(vals.shape, np.nan), where=vals > 0)


def col_batches(src: Iterable[Mapping[str, np.ndarray]], size: int = POINT_BATCH_SIZE) -> Iterator[PointCols]:
    """Combine small archive batches without retaining decoded theorem records.

    Concatenation preserves exact object integers; splitting keeps even one large
    incoming array from making an oversized pandas frame. Single chunks are
    borrowed, not copied; consumers retaining slices must account for their bases.
    """
    if size <= 0:
        raise ValueError("column batch size must be positive")
    chunks: list[Mapping[str, np.ndarray]] = []
    count = 0
    for cols in src:
        length = len(next(iter(cols.values())))
        start = 0
        while start < length:
            end = min(length, start + size - count)
            chunks.append(
                dict(cols) if start == 0 and end == length else {name: vals[start:end] for name, vals in cols.items()}
            )
            count += end - start
            start = end
            if count == size:
                yield (
                    dict(chunks[0])
                    if len(chunks) == 1
                    else {name: np.concatenate([chunk[name] for chunk in chunks]) for name in cols}
                )
                chunks.clear()
                count = 0
    if chunks:
        yield (
            dict(chunks[0])
            if len(chunks) == 1
            else {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
        )


def col_bytes(cols: Mapping[str, np.ndarray]) -> int:
    """Include referenced Python scalars in exceptional object-valued columns."""
    return sum(
        vals.nbytes + (sum(map(sys.getsizeof, vals.flat)) if vals.dtype.hasobject else 0) for vals in cols.values()
    )


# Raster aggregation and display.


def _point_cols(src: FrameSrc, plot: DensityPlot, cfg: DensityCfg) -> Iterator[PointCols]:
    cols = [plot.x.field, plot.y.field]
    cols.extend(field for field in (cfg.median, cfg.weight, cfg.colour) if field is not None)
    names = tuple(dict.fromkeys(cols))
    for frame in frame_batches(src):
        vals = {
            name: frame[name].to_numpy(copy=False) if isinstance(frame, pd.DataFrame) else frame[name] for name in names
        }
        length = len(vals[names[0]])
        # Do not rechunk/coalesce already batched sources. Bound unusually large
        # inputs here before masking, and detach views before retaining them so
        # the coordinate budget does not hide unrelated DataFrame/source buffers.
        for start in range(0, length, POINT_BATCH_SIZE):
            chunk = (
                vals
                if start == 0 and length <= POINT_BATCH_SIZE
                else {name: col[start : start + POINT_BATCH_SIZE] for name, col in vals.items()}
            )
            valid = np.ones(len(chunk[names[0]]), dtype=bool)
            for col in chunk.values():
                valid &= ~pd.isna(col) & (col != np.inf) & (col != -np.inf) if col.dtype.hasobject else np.isfinite(col)
            if valid.all():
                yield {name: col.copy() if col.base is not None else col for name, col in chunk.items()}
            elif valid.any():
                yield {name: col[valid] for name, col in chunk.items()}


def prepare_points(src: FrameSrc, plot: DensityPlot, cfg: DensityCfg, *, budget: int | None = None) -> PreparedPoints:
    """One source pass for valid, display-ready coordinates and their bounds.

    Drop retained columns (not observations) above the budget. The caller can
    then replay the source; no temporary persistence or sampling is introduced.
    """
    budget = POINT_MEMORY_BUDGET if budget is None else budget
    chunks: list[PointCols] | None = []
    nbytes = 0
    los, his = np.full(2, np.inf), np.full(2, -np.inf)
    for cols in _point_cols(src, plot, cfg):
        for idx, name in enumerate((plot.x.field, plot.y.field)):
            los[idx] = min(los[idx], float(cols[name].min()))
            his[idx] = max(his[idx], float(cols[name].max()))
        if chunks is not None:
            nbytes += col_bytes(cols)
            if nbytes > budget:
                chunks = None
                nbytes = 0
            else:
                chunks.append(cols)
    bounds = None
    if np.isfinite(los).all():
        pad = np.where(los == his, 0.1, (his - los) * 0.015)
        bounds = (float(los[0] - pad[0]), float(his[0] + pad[0])), (float(los[1] - pad[1]), float(his[1] + pad[1]))
    return PreparedPoints(chunks, bounds, nbytes)


def point_frames(
    src: FrameSrc, plot: DensityPlot, cfg: DensityCfg, prepared: PreparedPoints | None
) -> Iterator[pd.DataFrame]:
    chunks = (
        iter(prepared.chunks) if prepared is not None and prepared.chunks is not None else _point_cols(src, plot, cfg)
    )
    for cols in chunks:
        yield pd.DataFrame(cols, copy=False)


def prepare_point_raster(
    frame: pd.DataFrame, canvas: ds.Canvas, x: str, y: str, agg: ds.reductions.Reduction
) -> PointRaster:
    """Bind Datashader's native append kernels to persistent raster buffers.

    The pinned pandas pipeline creates/finalizes a complete grid per call to
    Canvas.points. Reuse its compiler and Point glyph instead: each input batch
    only visits its points, and xarray/grid finalization happens once. The glyph's
    private kernel binding is isolated here and checked against public points().
    """
    if canvas.x_range is None or canvas.y_range is None:
        raise ValueError("streamed point reduction requires fixed canvas ranges")
    glyph = Point(x, y)
    schema = dshape_from_pandas(frame).measure
    glyph.validate(schema)
    agg.validate(schema)
    canvas.validate()
    canvas.validate_ranges(canvas.x_range, canvas.y_range)
    create, info, append, _, finalize, antialias, antialias_fns, _ = cast(
        Callable[..., tuple[Any, ...]], compile_components
    )(agg, schema, glyph)
    # Datashader's private compiler exposes these native callables as object.
    create = cast(Callable[[tuple[int, int]], tuple[np.ndarray, ...]], create)
    finalize = cast("Callable[..., DataArray]", finalize)
    extend = cast(
        Callable[..., None],
        cast(Callable[..., Callable[..., None]], glyph._build_extend)(
            canvas.x_axis.mapper, canvas.y_axis.mapper, info, append, antialias, antialias_fns
        ),
    )
    bases = create((canvas.plot_height, canvas.plot_width))
    # Public count() uses uint32 per batch. Persistent buffers must not wrap
    # after many batches; the CPU append kernel specializes to uint64 too.
    bases = tuple(base.astype(np.uint64) if base.dtype == np.uint32 else base for base in bases)
    x_st = canvas.x_axis.compute_scale_and_translate(canvas.x_range, canvas.plot_width)
    y_st = canvas.y_axis.compute_scale_and_translate(canvas.y_range, canvas.plot_height)
    return PointRaster(
        partial(extend, bases, vt=x_st + y_st, bounds=canvas.x_range + canvas.y_range),
        partial(
            finalize,
            bases,
            cuda=False,
            coords={
                x: canvas.x_axis.compute_index(x_st, canvas.plot_width),
                y: canvas.y_axis.compute_index(y_st, canvas.plot_height),
            },
            dims=[y, x],
            attrs={"x_range": canvas.x_range, "y_range": canvas.y_range},
        ),
    )


def density_raster(
    src: FrameSrc, plot: DensityPlot, cfg: DensityCfg, canvas: ds.Canvas, *, prepared: PreparedPoints | None = None
) -> Raster:
    """Stream native reductions into one grid, then apply display scaling once.

    Exact medians retain only (pixel ID, value) numeric columns. They are not
    additive, so this is the intentional population-sized exception, not a sample.
    """
    shape = (canvas.plot_height, canvas.plot_width)
    agg = (
        ds.mean(cfg.colour)
        if cfg.colour and cfg.reduce_color == "mean"
        else ds.max(cfg.colour)
        if cfg.colour
        else ds.sum(cfg.weight)
        if cfg.weight
        else ds.count()
    )
    reduction: PointRaster | None = None
    median_cols: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in point_frames(src, plot, cfg, prepared):
        if cfg.median is not None:
            assert canvas.x_range is not None and canvas.y_range is not None
            # Counts and retained median values must describe the same population.
            frame = frame.loc[
                frame[plot.x.field].between(*canvas.x_range) & frame[plot.y.field].between(*canvas.y_range)
            ]
        if frame.empty:
            continue
        if reduction is None:
            reduction = prepare_point_raster(frame, canvas, plot.x.field, plot.y.field, agg)
        reduction.append(frame)
        if cfg.median is not None:
            assert canvas.x_range is not None and canvas.y_range is not None
            pxs = []
            for axis, mapper, bounds, resolution in (
                (plot.x, canvas.x_axis, canvas.x_range, shape[1]),
                (plot.y, canvas.y_axis, canvas.y_range, shape[0]),
            ):
                scale, translate = mapper.compute_scale_and_translate(bounds, resolution)
                # Use the Point glyph's scale/translate order and upper-edge
                # inclusion, not a separately rounded normalized coordinate.
                pixels = (mapper.mapper(frame[axis.field].to_numpy()) * scale + translate).astype(np.int32)
                pxs.append(np.minimum(pixels, resolution - 1))
            # Retain only the two numeric columns, not a DataFrame and its
            # index per theorem or a view keeping unrelated frame columns alive.
            median_cols.append((pxs[1] * shape[1] + pxs[0], frame[cfg.median].to_numpy(copy=True)))
    grid = reduction.finish().values if reduction is not None else np.zeros(shape)
    if cfg.colour is not None:
        return Raster(canvas, grid if reduction is not None else np.full(shape, np.nan), cfg.label or cfg.colour)
    if cfg.median is not None:
        vals = np.full(shape, np.nan)
        if median_cols:
            cols = pd.DataFrame(
                {
                    "px": np.concatenate([pxs for pxs, _ in median_cols]),
                    "val": np.concatenate([vals for _, vals in median_cols]),
                }
            )
            median_cols.clear()
            medians = cols.groupby("px")["val"].median()
            vals.ravel()[medians.index.to_numpy()] = medians.to_numpy()
        vals[grid < 3] = np.nan
        return Raster(canvas, vals, cfg.label or "Median largest-hypothesis fraction (bins ≥3 states)")
    img = np.where(grid > 0, np.log10(np.maximum(grid, np.finfo(float).tiny)), np.nan)
    label = "Weighted observations per bin (log scale)" if cfg.weight else "Observations per bin (log scale)"
    return Raster(canvas, img, label)


def _density_norm(img: np.ndarray, cfg: DensityCfg) -> Normalize:
    fin = img[np.isfinite(img)]
    lo, hi = cfg.colour_range or (float(fin.min()) if fin.size else 0.0, float(fin.max()) if fin.size else 1.0)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return Normalize(lo, hi)


def spread_px(img: np.ndarray, diam: int) -> np.ndarray:
    """Spread aggregate pixels; overlaps take a deterministic maximum."""
    if diam <= 1:
        return img
    coords = np.arange(diam) - (diam - 1) / 2
    disk = coords[:, None] ** 2 + coords[None, :] ** 2 <= (diam / 2) ** 2
    spread = max_filter(np.where(np.isfinite(img), img, -np.inf), footprint=disk, mode="constant", cval=-np.inf)
    spread[~np.isfinite(spread)] = np.nan
    return spread


# Figure builders.


def draw_graph(graph: Graph | GraphView, positions: PropertyMap, ax: Axes, style: GraphStyle) -> None:
    """One graph-tool adapter for shared atlas rendering conventions."""
    from graph_tool.draw import graph_draw

    markers: dict[str, Any] = {} if style.marker_size is None else {"edge_marker_size": style.marker_size}
    graph_draw(
        graph,
        pos=positions,
        vertex_fill_color=style.fill,
        vertex_size=style.size,
        edge_color=style.edges,
        edge_pen_width=style.edge_width,
        vertex_pen_width=0,
        mplfig=ax,
        fit_view=False,
        **markers,
    )


@plot_style
def density_figure(raster: Raster, plot: DensityPlot, cfg: DensityCfg, *, dot_diam: int = 1) -> Figure:
    """Build from a prepared raster; the caller owns saving and closing."""
    canvas, img, label = raster.canvas, raster.vals, raster.label
    assert canvas.x_range is not None and canvas.y_range is not None
    norm = _density_norm(img, cfg)
    img = spread_px(img, dot_diam)
    cmap = mpl.colormaps["turbo"].with_extremes(bad="black")
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.imshow(
        np.ma.masked_invalid(img),
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
        extent=(*canvas.x_range, *canvas.y_range),
        interpolation="nearest",
    )
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label)
    if cfg.median is None and (cfg.colour is None or cfg.colour_proj == "log10"):
        actual_val_ticks(colorbar.ax.yaxis)
    for axis, spec in ((ax.xaxis, plot.x), (ax.yaxis, plot.y)):
        format_axis(axis, spec.proj)
        axis.set_label_text(spec.label)
    ax.set_title(plot.title)
    caption = f"\n1400 × 1000 pixel grid; dot diameter {dot_diam} px; no interpolation."
    fig.text(0.1, 0.01, plot.caption + caption, fontsize=9)
    if plot.diag:
        lo, hi = (min(*canvas.x_range, *canvas.y_range), max(*canvas.x_range, *canvas.y_range))
        ax.plot([lo, hi], [lo, hi], "w--", alpha=0.5)
        ax.set_xlim(canvas.x_range)
        ax.set_ylim(canvas.y_range)
    return fig


@plot_style
def panel_figure(
    imgs: list[np.ndarray],
    xb: tuple[float, float],
    yb: tuple[float, float],
    titles: Sequence[str],
    cfg: PanelCfg,
    *,
    dot_diam: int,
) -> Figure:
    """Prepared grids share bounds and colour scale; no source/archive access."""
    max_ = max((float(np.nanmax(img)) for img in imgs if np.isfinite(img).any()), default=0)
    norm = Normalize(0, max(1, max_))
    cmap = plt.colormaps["turbo"].with_extremes(bad="black")
    fig, axes = plt.subplots(1, len(imgs), figsize=(7 * len(imgs), 6.5), squeeze=False, sharex=True, sharey=True)
    for ax, img, title in zip(axes.flat, imgs, titles, strict=True):
        img = spread_px(img, dot_diam)
        ax.imshow(img, extent=(*xb, *yb), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        if cfg.diag:
            # An equality reference line does not require matching axis ranges:
            # DAG size and expanded size can differ by many orders of magnitude.
            lo, hi = max(xb[0], yb[0]), min(xb[1], yb[1])
            if lo < hi:
                ax.plot([lo, hi], [lo, hi], "w--", alpha=0.5)
        ax.set(title=title, xlabel=cfg.xlabel, xlim=xb, ylim=yb)
        format_axis(ax.xaxis, cfg.x_proj)
        format_axis(ax.yaxis, cfg.y_proj)
    axes[0, 0].set_ylabel(cfg.ylabel)
    label = "Weighted observations per pixel (log scale)"
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), label=label, fraction=0.025)
    actual_val_ticks(colorbar.ax.yaxis)
    caption_default = "Same original observations, axis bounds and colour scale in every panel.\nLogarithmic axes except percentages; node counts exclude retained binder records."
    fig.text(0.08, 0.01, cfg.caption or caption_default, fontsize=9)
    fig.subplots_adjust(bottom=0.16, top=0.90, right=0.88, wspace=0.12)
    return fig


# Saving and drawing entry points.


def save_fig(fig: Figure, path: Path, *, dpi: int = 130, tight: bool = False) -> None:
    """All renderers share the save/close boundary; failed saves also close."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None)
    finally:
        plt.close(fig)


def save_single(fig: Figure, path: Path) -> None:
    """Single-panel figures fit a 1080p screen; atlases retain their own sizes."""
    save_fig(fig, path, dpi=110, tight=True)


def density(
    dir: Path,
    frame: FrameSrc,
    plot: DensityPlot,
    cfg: DensityCfg = DensityCfg(),  # noqa: B008
    *,
    dot_diam: int = 1,
    prepared: PreparedPoints | None = None,
) -> None:
    """Aggregate observations once, then draw a single fixed-resolution raster."""
    prepared = prepared if prepared is not None else prepare_points(frame, plot, cfg)
    bounds = prepared.bounds
    if bounds is None:
        return
    raster = density_raster(
        frame,
        plot,
        cfg,
        ds.Canvas(plot_width=1400, plot_height=1000, x_range=bounds[0], y_range=bounds[1]),
        prepared=prepared,
    )
    save_single(density_figure(raster, plot, cfg, dot_diam=dot_diam), dir / plot.filename)


def density_panels(
    frames: Sequence[FrameSrc], titles: Sequence[str], path: Path, panel_cfg: PanelCfg, *, dot_diam: int
) -> list[PreparedPoints]:
    """Comparable fixed pixel grids: identical bounds and colour scales per panel."""

    plot = DensityPlot(
        "", "", Axis("x", panel_cfg.xlabel, panel_cfg.x_proj), Axis("y", panel_cfg.ylabel, panel_cfg.y_proj), ""
    )
    density_cfg = DensityCfg(weight="weight")
    prepared: list[PreparedPoints] = []
    remaining = POINT_MEMORY_BUDGET
    for frame in frames:
        points = prepare_points(frame, plot, density_cfg, budget=remaining)
        prepared.append(points)
        remaining -= points.nbytes
    ranges = [points.bounds for points in prepared if points.bounds is not None]
    if not ranges:
        return prepared
    xb, yb = tuple((min(bounds[i][0] for bounds in ranges), max(bounds[i][1] for bounds in ranges)) for i in (0, 1))
    if panel_cfg.eq_ranges:
        xb = yb = (min(xb[0], yb[0]), max(xb[1], yb[1]))
    canvas = ds.Canvas(plot_width=1000, plot_height=800, x_range=xb, y_range=yb)
    imgs = [
        density_raster(frame, plot, density_cfg, canvas, prepared=points).vals
        for frame, points in zip(frames, prepared, strict=True)
    ]
    save_fig(panel_figure(imgs, xb, yb, titles, panel_cfg, dot_diam=dot_diam), path, tight=True)
    return prepared
