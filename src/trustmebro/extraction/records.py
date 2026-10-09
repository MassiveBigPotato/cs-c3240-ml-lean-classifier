"""Typed theorem records parsed from the NDJSON emitted by ``Extract.lean``.

The Lean executable writes one self-contained theorem per line.  This module
keeps parsing independent from the eventual storage codec so the same checked
records can later be encoded as MessagePack and written to SQLite.

msgspec validates fixed-shape fields, while constructor dispatch and
cross-reference checks remain explicit. Parse failures raise SchemaError.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from functools import cache
from typing import Annotated, BinaryIO, TypeAliasType

import msgspec

type Nat = Annotated[int, msgspec.Meta(ge=0)]
type PosNat = Annotated[int, msgspec.Meta(gt=0)]
type NonemptyString = Annotated[str, msgspec.Meta(min_length=1)]
type ExprArgs = Annotated[tuple[Nat, ...], msgspec.Meta(min_length=1)]
type BinderNames = Annotated[tuple[str, ...], msgspec.Meta(min_length=1)]
type Span = tuple[Nat, Nat]
type Parser[T] = Callable[[object, str], T]


class SchemaError(ValueError):
    """An input value does not conform to the extractor schema."""


class BinderInfo(StrEnum):
    DEFAULT = "default"
    IMPLICIT = "implicit"
    STRICT_IMPLICIT = "strictImplicit"
    INST_IMPLICIT = "instImplicit"


class LocalDeclKind(StrEnum):
    DEFAULT = "default"
    IMPL_DETAIL = "implementationDetail"
    AUXILIARY = "auxiliary"


class MvarKind(StrEnum):
    NATURAL = "natural"
    SYNTH = "synthetic"
    SYNTH_OPAQUE = "syntheticOpaque"


class _Array(msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True):
    """Canonical theorem data; positional when encoded as MessagePack."""


class _Variant(_Array, frozen=True, tag_field="_tag"):
    """Stable integer discriminator for alternatives in a typed union."""


# Tags are part of the binary representation; do not renumber persisted data.
class LvlZero(_Variant, frozen=True, tag=0):
    pass


class LvlMvar(_Variant, frozen=True, tag=1):
    id: Nat


class LvlSucc(_Variant, frozen=True, tag=2):
    level: Lvl


class LvlMax(_Variant, frozen=True, tag=3):
    left: Lvl
    right: Lvl


class LvlIMax(_Variant, frozen=True, tag=4):
    left: Lvl
    right: Lvl


class LvlParam(_Variant, frozen=True, tag=5):
    name: str


type Lvl = LvlZero | LvlMvar | LvlSucc | LvlMax | LvlIMax | LvlParam


class Bvar(_Variant, frozen=True, tag=0):
    idx: Nat


class Fvar(_Variant, frozen=True, tag=1):
    id: Nat


class ExprMvar(_Variant, frozen=True, tag=2):
    id: Nat


class Sort(_Variant, frozen=True, tag=3):
    lvl: Lvl


class Const(_Variant, frozen=True, tag=4):
    name: str
    universes: tuple[Lvl, ...]


class App(_Variant, frozen=True, tag=5):
    fn: Nat
    args: ExprArgs


class Lambda(_Variant, frozen=True, tag=6):
    names: BinderNames
    type: Nat
    body: Nat
    binder_info: BinderInfo


class Forall(_Variant, frozen=True, tag=7):
    names: BinderNames
    type: Nat
    body: Nat
    binder_info: BinderInfo


class Let(_Variant, frozen=True, tag=8):
    name: str
    type: Nat
    val: Nat
    body: Nat
    nondep: bool


class NatLiteral(_Variant, frozen=True, tag=9):
    val: Nat


class StringLiteral(_Variant, frozen=True, tag=10):
    val: str


class Metadata(_Variant, frozen=True, tag=11):
    data: tuple[tuple[str, str], ...]
    expr: Nat


class Proj(_Variant, frozen=True, tag=12):
    type_name: str
    idx: Nat
    struct: Nat


type Expr = (
    Bvar | Fvar | ExprMvar | Sort | Const | App | Lambda | Forall | Let | NatLiteral | StringLiteral | Metadata | Proj
)


class LocalConst(_Variant, frozen=True, tag=0):
    id: Nat
    type: Nat
    kind: LocalDeclKind
    is_instance: bool
    binder_info: BinderInfo


class LocalLet(_Variant, frozen=True, tag=1):
    id: Nat
    type: Nat
    kind: LocalDeclKind
    is_instance: bool
    val: Nat
    nondep: bool


type LocalDecl = LocalConst | LocalLet


class MvarDeclr(_Array, frozen=True):
    type: Nat
    locals: tuple[LocalDecl, ...]
    kind: MvarKind


class DelayedAssign(_Array, frozen=True):
    fvars: tuple[Nat, ...]
    pending: Nat


class MvarState(_Array, frozen=True):
    id: Nat
    decl: MvarDeclr | None
    assignment: Nat | None
    delayed_assign: DelayedAssign | None


class LvlMvarDecl(_Array, frozen=True):
    depth: Nat
    idx: Nat


class LvlMvarState(_Array, frozen=True):
    id: Nat
    decl: LvlMvarDecl | None
    assignment: Lvl | None


class ProofState(_Array, frozen=True):
    target: Nat
    locals: tuple[LocalDecl, ...]
    mvars: tuple[MvarState, ...]
    lvl_mvars: tuple[LvlMvarState, ...]


class Tactic(_Array, frozen=True):
    kind: str
    src: str


class Trn(_Array, frozen=True):
    tactic: Tactic
    src_span: Span
    open_goal_count: PosNat
    state: ProofState


class Theorem(_Array, frozen=True):
    name: NonemptyString
    module: NonemptyString
    src_span: Span | None
    exprs: tuple[Expr, ...]
    trns: Annotated[tuple[Trn, ...], msgspec.Meta(min_length=1)]


_json_decoder = msgspec.json.Decoder()


def _schema_err(path: str, msg: str) -> SchemaError:
    return SchemaError(f"{path}: {msg}")


def _decode[T](val: object, schema: type[T] | TypeAliasType, path: str) -> T:
    try:
        return msgspec.convert(val, type=schema, strict=True)
    except msgspec.ValidationError as err:
        raise _schema_err(path, str(err)) from err


def _object(val: object, path: str) -> dict[str, object]:
    if not isinstance(val, dict) or not all(isinstance(key, str) for key in val):
        raise _schema_err(path, "expected an object")
    return val


@cache
def _array_of[T](parser: Parser[T]) -> Parser[tuple[T, ...]]:
    """Adapt an element parser to a JSON array, retaining indexed error paths."""

    def parse(val: object, path: str) -> tuple[T, ...]:
        if not isinstance(val, list):
            raise _schema_err(path, "expected an array")
        return tuple(parser(item, f"{path}[{idx}]") for idx, item in enumerate(val))

    return parse


@cache
def _optional_of[T](parser: Parser[T]) -> Parser[T | None]:
    """JSON null represents an absent value, not a value to decode."""

    def parse(val: object, path: str) -> T | None:
        return None if val is None else parser(val, path)

    return parse


def _json_name(field: str) -> str:
    head, *tail = field.split("_")
    return head + "".join(part.capitalize() for part in tail)


@cache
def _record_layout(cls: type[msgspec.Struct]) -> tuple[tuple[tuple[str, str], ...], frozenset[str], str | int | None]:
    fields = tuple((name, _json_name(name)) for name in cls.__struct_fields__)
    return fields, frozenset(key for _, key in fields), cls.__struct_config__.tag


def _record[T: msgspec.Struct](
    val: object, cls: type[T], path: str, transforms: Mapping[str, Parser[object]] | None = None
) -> T:
    """Reorder a Lean JSON object into its one canonical positional schema."""
    fields = _object(val, path)
    layout, keys, tag = _record_layout(cls)
    if fields.keys() != keys:
        missing, extra = keys - fields.keys(), fields.keys() - keys
        problems: list[str] = []
        if missing:
            problems.append(f"missing fields {sorted(missing)}")
        if extra:
            problems.append(f"unknown fields {sorted(extra)}")
        raise _schema_err(path, "; ".join(problems))
    ordered: list[object] = [tag] if tag is not None else []
    for name, key in layout:
        item = fields[key]
        if transforms is not None and name in transforms:
            item = transforms[name](item, f"{path}.{key}")
        ordered.append(item)
    return _decode(ordered, cls, path)


@cache
def _record_of[T: msgspec.Struct](cls: type[T], **transforms: Parser[object]) -> Parser[T]:
    """Use a record schema and its field parsers without an inline lambda."""

    def parse(val: object, path: str) -> T:
        return _record(val, cls, path, transforms)

    return parse


def _variant(val: object, path: str, name: str) -> tuple[str, object]:
    record = _object(val, path)
    if len(record) != 1:
        raise _schema_err(path, f"expected exactly one {name} constructor")
    return next(iter(record.items()))


def _span(span: Span, path: str) -> Span:
    if span[1] < span[0]:
        raise _schema_err(path, "stop offset precedes start offset")
    return span


def _parse_lvl(val: object, path: str) -> Lvl:
    if val == "zero":
        return LvlZero()
    kind, data = _variant(val, path, "level")
    child_path = f"{path}.{kind}"
    match kind:
        case "mvar":
            return LvlMvar(_decode(data, Nat, child_path))
        case "succ":
            return LvlSucc(_parse_lvl(data, child_path))
        case "max" | "imax":
            left, right = _decode(data, tuple[object, object], child_path)
            pair = (_parse_lvl(left, f"{child_path}[0]"), _parse_lvl(right, f"{child_path}[1]"))
            return LvlMax(*pair) if kind == "max" else LvlIMax(*pair)
        case "param":
            return LvlParam(_decode(data, str, child_path))
        case _:
            raise _schema_err(path, f"unknown level constructor {kind!r}")


def _parse_metadata(val: object, path: str) -> tuple[tuple[str, str], ...]:
    return tuple(_decode(val, dict[str, str], path).items())


def _parse_expr(val: object, path: str) -> Expr:
    kind, data = _variant(val, path, "expression")
    child_path = f"{path}.{kind}"
    match kind:
        case "bvar":
            return Bvar(_decode(data, Nat, child_path))
        case "fvar":
            return Fvar(_decode(data, Nat, child_path))
        case "mvar":
            return ExprMvar(_decode(data, Nat, child_path))
        case "sort":
            return Sort(_parse_lvl(data, child_path))
        case "const":
            return _record(data, Const, child_path, {"universes": _array_of(_parse_lvl)})
        case "app":
            return _record(data, App, child_path)
        case "lambda":
            return _record(data, Lambda, child_path)
        case "forall":
            return _record(data, Forall, child_path)
        case "let":
            return _record(data, Let, child_path)
        case "natural":
            return NatLiteral(_decode(data, Nat, child_path))
        case "string":
            return StringLiteral(_decode(data, str, child_path))
        case "metadata":
            return _record(data, Metadata, child_path, {"data": _parse_metadata})
        case "projection":
            return _record(data, Proj, child_path)
        case _:
            raise _schema_err(path, f"unknown expression constructor {kind!r}")


def _parse_local_decl(val: object, path: str) -> LocalDecl:
    cls = LocalConst if "binderInfo" in _object(val, path) else LocalLet
    return _record(val, cls, path)


def _parse_mvar_state(val: object, path: str) -> MvarState:
    return _record(
        val,
        MvarState,
        path,
        {
            "decl": _optional_of(_record_of(MvarDeclr, locals=_array_of(_parse_local_decl))),
            "delayed_assign": _optional_of(_record_of(DelayedAssign)),
        },
    )


def _parse_lvl_mvar_state(val: object, path: str) -> LvlMvarState:
    return _record(
        val, LvlMvarState, path, {"decl": _optional_of(_record_of(LvlMvarDecl)), "assignment": _optional_of(_parse_lvl)}
    )


def _parse_proof_state(val: object, path: str) -> ProofState:
    return _record(
        val,
        ProofState,
        path,
        {
            "locals": _array_of(_parse_local_decl),
            "mvars": _array_of(_parse_mvar_state),
            "lvl_mvars": _array_of(_parse_lvl_mvar_state),
        },
    )


def expr_refs(expr: Expr) -> tuple[int, ...]:
    """Ordered structural operands, including repeats; no assignment or unfolding edges."""
    match expr:
        case App(fn, args):
            return fn, *args
        case Lambda(type=type_idx, body=body) | Forall(type=type_idx, body=body):
            return type_idx, body
        case Let(type=type_idx, val=val, body=body):
            return type_idx, val, body
        case Metadata(expr=inner) | Proj(struct=inner):
            return (inner,)
        case _:
            return ()


def _local_expr_refs(decl: LocalDecl) -> tuple[int, ...]:
    if isinstance(decl, LocalLet):
        return decl.type, decl.val
    return (decl.type,)


def _state_expr_refs(state: ProofState) -> Iterator[int]:
    yield state.target
    for decl in state.locals:
        yield from _local_expr_refs(decl)
    for mvar in state.mvars:
        if mvar.assignment is not None:
            yield mvar.assignment
        if mvar.delayed_assign is not None:
            yield from mvar.delayed_assign.fvars
        if mvar.decl is not None:
            yield mvar.decl.type
            for decl in mvar.decl.locals:
                yield from _local_expr_refs(decl)


def _validate_refs(theorem: Theorem) -> None:
    for idx, expr in enumerate(theorem.exprs):
        for ref in expr_refs(expr):
            if ref >= idx:
                raise _schema_err(f"$.exprs[{idx}]", f"expr ref {ref} is not earlier in the post-order table")
    n_exprs = len(theorem.exprs)
    for trn_idx, trn in enumerate(theorem.trns):
        for ref in _state_expr_refs(trn.state):
            if ref >= n_exprs:
                raise _schema_err(f"$.trns[{trn_idx}].state", f"expr ref {ref} is outside a table of size {n_exprs}")


def validate_theorem(theorem: Theorem) -> Theorem:
    """Check constraints spanning fields after a record has been decoded."""
    if theorem.src_span is not None:
        _span(theorem.src_span, "$.srcSpan")
    for idx, trn in enumerate(theorem.trns):
        _span(trn.src_span, f"$.trns[{idx}].srcSpan")
    _validate_refs(theorem)
    return theorem


def parse_theorem(val: object) -> Theorem:
    """Validate and convert one already-decoded theorem object.

    Raises:
        SchemaError: If any part of the object violates the extractor schema.
    """

    theorem = _record(
        val,
        Theorem,
        "$",
        {
            "exprs": _array_of(_parse_expr),
            "trns": _array_of(_record_of(Trn, tactic=_record_of(Tactic), state=_parse_proof_state)),
        },
    )
    return validate_theorem(theorem)


def iter_theorems(stream: BinaryIO) -> Iterator[Theorem]:
    """Parse a binary NDJSON stream one theorem at a time.

    Raises:
        SchemaError: During iteration, if a line is not valid extractor output.
    """

    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            raise SchemaError(f"line {line_number}: blank lines are not valid extractor output")
        try:
            value = _json_decoder.decode(line)
        except msgspec.DecodeError as error:
            raise SchemaError(f"line {line_number}: invalid JSON: {error}") from error
        try:
            theorem = parse_theorem(value)
        except SchemaError as error:
            raise SchemaError(f"line {line_number}: {error}") from error
        yield theorem
