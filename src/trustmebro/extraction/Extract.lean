import Lake.Load.Workspace

/-!
Extract source-level tactic trns and their elaborated pre-tactic states.

The per-file elaboration and `InfoTree` traversal are adapted from
LeanDojo-v2's `ExtractData.lean` (MIT). The interned expression format is
adapted from lean4export (Apache-2.0), extended to retain open expressions.

Selection happens before serialization: discarded `InfoTree` nodes are never
pretty-printed and their expressions are never traversed by the encoder.
-/

open Lean Elab System
open Std (HashMap HashSet)

namespace Trustmebro.Extraction

/- Error messages defined here to keep them out of control flow.
Makes it much easier to follow. -/
local macro "lakeEnvErr" err:ident : term => `(IO.userError s!"Could not configure the Lake environment: {$err}")
local macro "lakeInstallErr" : term => `(IO.userError "Could not locate the Lake installation")
local macro "leanInstallErr" : term => `(IO.userError "Could not locate the Lean installation")
local macro "lakeLoadErr" : term => `(IO.userError "Could not load the Lake workspace")
local macro "notInModuleErr " path:ident : term => `(IO.userError s!"No lake module contains source file: {$path}")
local macro "ctxInfoSynthErr" : term => `(IO.userError "Could not synthesize ContextInfo while traversing InfoTree")
local macro "topLvlCtxSynthErr" : term => `(IO.userError "Could not synthesize top-level ContextInfo")


structure ExtractionTarget where
  path : FilePath
  moduleName : Name
  options : Options

private def loadWorkspace : IO Lake.Workspace := do
  let cwd ← IO.currentDir
  let (elanInstall?, leanInstall?, lakeInstall?) ← Lake.findInstall?
  let some leanInstall := leanInstall? | throw leanInstallErr
  let some lakeInstall := lakeInstall? | throw lakeInstallErr
  let lakeEnv ← Lake.Env.compute lakeInstall leanInstall elanInstall?
    |>.toIO fun err => lakeEnvErr err
  let loadConfig : Lake.LoadConfig := {
    lakeEnv
    wsDir := cwd
    updateToolchain := false
  }
  let some workspace ← (Lake.loadWorkspace loadConfig).toBaseIO | throw <| lakeLoadErr
  return workspace

private def resolveExtractionTarget
  (workspace : Lake.Workspace) (path : FilePath) : IO ExtractionTarget := do
  let path ← IO.FS.realPath path
  let some module := workspace.findModuleBySrc? path | throw <| notInModuleErr path
  return {
    path
    moduleName := module.name
    options := module.leanOptions.toOptions
  }

/-! ## Transition selection -/

structure SourceKey where
  kind : Name
  start : String.Pos.Raw
  stop : String.Pos.Raw
deriving BEq, Hashable

abbrev SourceIndex := HashSet SourceKey

private def sourceKey? (stx : Syntax) : Option SourceKey := do
  let start ← stx.getPos? (canonicalOnly := true)
  let stop ← stx.getTailPos? (canonicalOnly := true)
  guard (start < stop)
  return {kind := stx.getKind, start, stop}

private partial def indexSyntax (stx : Syntax) (index : SourceIndex) : SourceIndex :=
  let index := (sourceKey? stx).elim index index.insert
  stx.getArgs.foldl (fun index child => indexSyntax child index) index

private def rootCommandSyntax? : InfoTree → Option Syntax
  | .context _ tree => rootCommandSyntax? tree
  | .node (.ofCommandInfo info) _ => some info.stx
  | _ => none

/-- Index syntax from the parsed source commands, independently of tactic
elaboration. Requiring an elaborated tactic node to occur in this index keeps
macro-generated implementation tactics out of the exported label stream. -/
private def buildSourceIndex (trees : Array InfoTree) : SourceIndex :=
  trees.filterMap rootCommandSyntax? |>.foldl (fun index stx => indexSyntax stx index) {}

structure RawTrn where
  info : TacticInfo
  source : SourceKey

structure CollectionState where
  -- The separate order array avoids depending on hash-map iteration order.
  order : Array Name := #[]
  trns : HashMap Name (Array RawTrn) := {}

abbrev CollectM := StateT CollectionState IO

-- These generated syntax-kind names are not all addressable with declaration
-- name quotations, so construct the exceptional ones explicitly.
private def andThenKind : Name := .str `Lean.Parser.Tactic "tactic_<;>_"
private def semicolonKind : Name := .str .anonymous ";"
private def nextKind : Name := .str `Lean.Parser.Tactic "tacticNext_=>_"
private def solveKind : Name := .str `Lean "solveTactic"

/-- Sequence nodes directly contain the source-level tactics we want to treat
as individual model decisions. -/
private def sequenceKinds : HashSet Name := HashSet.ofArray #[
  ``Lean.Parser.Tactic.tacticSeq1Indented,
  ``Lean.Parser.Tactic.tacticSeqBracketed
]

/-- Nodes which organize or repeatedly execute source tactics, but are not
themselves model decisions. Their executed children remain eligible. -/
private def transparentKinds : HashSet Name := HashSet.ofArray #[
  andThenKind,
  semicolonKind,
  `Lean.Parser.Tactic.tacticRepeat_,
  `Lean.Parser.Tactic.repeat',
  `Lean.Parser.Tactic.repeat1',
  `Lean.Parser.Tactic.first,
  `Lean.Parser.Tactic.solve,
  solveKind,
  `Lean.Parser.Tactic.tacticTry_,
  ``Lean.Parser.Tactic.allGoals,
  ``Lean.Parser.Tactic.anyGoals,
  ``Lean.Parser.Tactic.focus,
  ``Lean.Parser.Tactic.paren,
  `Lean.cdot,
  ``Lean.Parser.Tactic.case,
  `Lean.Parser.Tactic.case',
  `Lean.Parser.Tactic.next,
  nextKind,
  `Mathlib.Tactic.failIfNoProgress
]

/-- These wrappers may retain failed attempts in their information trees. -/
private def rollbackFilteringKinds : HashSet Name := HashSet.ofArray #[
  `Lean.Parser.Tactic.tacticRepeat_,
  `Lean.Parser.Tactic.repeat',
  `Lean.Parser.Tactic.repeat1',
  ``Lean.Parser.Tactic.anyGoals
]

/-- `LocalDecl` has no `BEq` instance in Lean 4.34, so compare its fields
explicitly when checking whether a combinator child was rolled back. -/
private def localDeclEq : LocalDecl → LocalDecl → Bool
  | .cdecl i₁ f₁ n₁ t₁ b₁ k₁, .cdecl i₂ f₂ n₂ t₂ b₂ k₂ =>
    i₁ == i₂ && f₁ == f₂ && n₁ == n₂ && t₁ == t₂ && b₁ == b₂ && k₁ == k₂
  | .ldecl i₁ f₁ n₁ t₁ v₁ d₁ k₁, .ldecl i₂ f₂ n₂ t₂ v₂ d₂ k₂ =>
    i₁ == i₂ && f₁ == f₂ && n₁ == n₂ && t₁ == t₂ && v₁ == v₂ && d₁ == d₂ && k₁ == k₂
  | _, _ => false

private def localDeclListEq : List (Option LocalDecl) → List (Option LocalDecl) → Bool
  | [], [] => true
  | none :: left, none :: right => localDeclListEq left right
  | some leftDecl :: left, some rightDecl :: right => localDeclEq leftDecl rightDecl && localDeclListEq left right
  | _, _ => false

private def localContextEq (left right : LocalContext) : Bool :=
  localDeclListEq left.decls.toList right.decls.toList

private def metavarKindEq : MetavarKind → MetavarKind → Bool
  | .natural, .natural | .synthetic, .synthetic | .syntheticOpaque, .syntheticOpaque => true
  | _, _ => false

private def goalDeclEq (left right : MetavarContext) (goal : MVarId) : Bool :=
  match left.findDecl? goal, right.findDecl? goal with
  | some leftDecl, some rightDecl => leftDecl.type == rightDecl.type
    && localContextEq leftDecl.lctx rightDecl.lctx
    && leftDecl.localInstances == rightDecl.localInstances
    && metavarKindEq leftDecl.kind rightDecl.kind
  | none, none => true
  | _, _ => false

private def goalViewEq (left right : MetavarContext) (goal : MVarId) : Bool :=
  left.getExprAssignmentCore? goal == right.getExprAssignmentCore? goal && goalDeclEq left right goal

private def tacticMadeProgress (info : TacticInfo) : Bool :=
  info.goalsBefore != info.goalsAfter || !info.goalsBefore.all (goalViewEq info.mctxBefore info.mctxAfter)

/-- Whether a child result is present in its wrapper's committed final state.
This rejects a partially executed child whose exception was caught and rolled back. -/
private def childWasCommitted (wrapper child : TacticInfo) : Bool :=
  child.goalsBefore.all (goalViewEq child.mctxAfter wrapper.mctxAfter)

private def rootTacticInfo? : InfoTree → Option TacticInfo
  | .context _ tree => rootTacticInfo? tree
  | .node (.ofTacticInfo info) _ => some info
  | _ => none

private def addTransition (declaration : Name) (info : TacticInfo) (source : SourceKey) : CollectM Unit := do
  let state ← get
  let isNewDeclaration := !state.trns.contains declaration
  let trns := state.trns.getD declaration #[] |>.push {info, source}
  set {state with
    order := if isNewDeclaration then state.order.push declaration else state.order
    trns := state.trns.insert declaration trns
  }

/-- Select trns while walking the elaborator tree.
A transition must be an executed child of a tactic sequence, must not be a structural combinator,
and must match syntax independently indexed from the source command.
The latter condition rejects implementation tactics introduced only by macros. -/
private partial def collectTree
  (sources : SourceIndex)
  (ctx : ContextInfo)
  (tree : InfoTree)
  (parentTactic? : Option TacticInfo) :
  CollectM Unit := do
  match tree with
  | .context innerCtx innerTree =>
    -- Context nodes store deltas, so we merge them to recover the context
    -- required to identify the surrounding declaration.
    let some merged := innerCtx.mergeIntoOuter? ctx | throw ctxInfoSynthErr
    collectTree sources merged innerTree parentTactic?
  | .node (.ofTacticInfo tacticInfo) children =>
    collectTacticNode tacticInfo children
  | .node _ children =>
    for child in children do collectTree sources ctx child parentTactic?
  | _ => pure ()
where
  collectTacticNode (tacticInfo : TacticInfo) (children : PersistentArray InfoTree) : CollectM Unit := do
    let kind := tacticInfo.stx.getKind
    let isBoundary := (parentTactic?.map (·.stx.getKind)).any sequenceKinds.contains

    if isBoundary && !tacticInfo.goalsBefore.isEmpty && !transparentKinds.contains kind then
      let some source := sourceKey? tacticInfo.stx |>.filter sources.contains | pure ()
      let some declaration := ctx.parentDecl? | pure ()
      addTransition declaration tacticInfo source

    if kind == ``Lean.Parser.Tactic.first then
      let some child := children.toArray.back? | return ()
      collectTree sources ctx child (some tacticInfo)
      return ()
    else if rollbackFilteringKinds.contains kind then
      for child in children do
        let some childInfo := rootTacticInfo? child | continue
        if tacticMadeProgress childInfo && childWasCommitted tacticInfo childInfo then
          collectTree sources ctx child (some tacticInfo)
      return ()
    else for child in children do
      collectTree sources ctx child (some tacticInfo)

private def collectTopLevelTree (sources : SourceIndex) (tree : InfoTree) : CollectM Unit := do
  let .context ctx innerTree := tree | pure ()
  let some ctx := ctx.mergeIntoOuter? none | throw topLvlCtxSynthErr
  collectTree sources ctx innerTree none

private def collectTransitions (sources : SourceIndex) (trees : Array InfoTree) : IO CollectionState := do
  let (_, state) ← (trees.forM (collectTopLevelTree sources)).run {}
  return state


/-! ## Flat expression records and representation rewrites -/

/-- Child fields are theorem-local record IDs, not recursive expression trees.
Interning these shallow keys preserves sharing without repeatedly comparing
deep normalized DAGs. Metadata retains its original structural key. -/
private inductive FlatExpr where
  | bvar : Nat → FlatExpr
  | fvar : Nat → FlatExpr
  | mvar : Nat → FlatExpr
  | sort : Level → FlatExpr
  | const : Name → List Level → FlatExpr
  | app (fn : Nat) (args : Array Nat)
  | lam : Array Name → Nat → Nat → BinderInfo → FlatExpr
  | forallE : Array Name → Nat → Nat → BinderInfo → FlatExpr
  | letE (declName : Name) (type : Nat) (value : Nat) (body : Nat) (nondep : Bool)
  | lit : Literal → FlatExpr
  | mdata (src : ExprStructEq) (expr : Nat)
  | proj (typeName : Name) (idx : Nat) (struct : Nat)
deriving BEq, Hashable

/-- Lossy operator families, with exact source arity including implicit args.
The result names denote feature families, not callable Lean applications. -/
private def operatorRule? : Name → Option (Name × Nat)
  | `Nat.add | `Int.add | `Rat.add | `Float.add => some (`Add.add, 2)
  | `Fin.add | `BitVec.add => some (`Add.add, 3)
  | `HAdd.hAdd => some (`Add.add, 6)
  | `Nat.mul | `Int.mul | `Rat.mul | `Float.mul => some (`Mul.mul, 2)
  | `Fin.mul | `BitVec.mul => some (`Mul.mul, 3)
  | `Nat.sub | `Int.sub | `Rat.sub | `Float.sub => some (`Sub.sub, 2)
  | `Fin.sub | `BitVec.sub => some (`Sub.sub, 3)
  | `Nat.div | `Int.ediv | `Rat.div | `Float.div => some (`Div.div, 2)
  | `Fin.div | `BitVec.udiv => some (`Div.div, 3)
  | _ => none

/-- Full projection arity and value slot. CoeT/CoeDep bind the coerced value
as a class parameter; their final argument is the instance, not the value. -/
private def coercionRule? : Name → Option (Nat × Nat)
  | `Coe.coe | `CoeOut.coe | `CoeTail.coe | `CoeHead.coe
  | `CoeFun.coe | `CoeSort.coe | `CoeTC.coe | `CoeOTC.coe
  | `CoeHTC.coe | `CoeHTCT.coe => some (4, 3)
  | `CoeT.coe | `CoeDep.coe => some (4, 1)
  | `Nat.cast | `Int.cast | `NatCast.natCast | `IntCast.intCast => some (3, 2)
  | _ => none

/-- Application preparation distinguishes existing expression heads from
synthetic operator-family names, which are interned without recursive encoding. -/
private inductive AppPlan where
  | application (fn : Expr) (args : Array Expr)
  | operator (family : Name) (operands : Array Expr)

/-- Select a representation rewrite without encoding any discarded arguments.
Coercion erasure retains the value and trailing applications, not the conversion
instance; this is deliberately lossy, not a semantics-preserving Lean rewrite. -/
private def appPlan (fn : Expr) (args : Array Expr) : AppPlan := Id.run do
  if let .const name _ := fn then
    if let some (dst, arity) := operatorRule? name then
      if args.size == arity then
        return .operator dst #[args[arity - 2]!, args[arity - 1]!]
    if let some (arity, slot) := coercionRule? name then
      if args.size >= arity then
        let value := args[slot]!
        let trailing := args.extract arity args.size
        return .application value trailing
  return .application fn args

/-- Only group domains that agree after lifting into the later binder's scope. -/
private def binderGroup (expr : Expr) : Array Name × Expr × Expr × BinderInfo := Id.run do
  let type := expr.bindingDomain!
  let info := expr.binderInfo
  let mut names := #[expr.bindingName!]
  let mut body := expr.bindingBody!
  while (if expr.isLambda then body.isLambda else body.isForall) do
    let domain := type.liftLooseBVars 0 names.size
    unless Expr.equal body.bindingDomain! domain && body.binderInfo == info do break
    names := names.push body.bindingName!
    body := body.bindingBody!
  return (names, type, body, info)


/-! ## Per-theorem expression encoding -/

private def binderInfoJson : BinderInfo → Json
  | .default => "default"
  | .implicit => "implicit"
  | .strictImplicit => "strictImplicit"
  | .instImplicit => "instImplicit"

private def localDeclKindJson : LocalDeclKind → Json
  | .default => "default"
  | .implDetail => "implementationDetail"
  | .auxDecl => "auxiliary"

private def metavarKindJson : MetavarKind → Json
  | .natural => "natural"
  | .synthetic => "synthetic"
  | .syntheticOpaque => "syntheticOpaque"

private def kvMapJson (kvs : KVMap) : Json :=
  .mkObj <| kvs.entries.map fun (key, value) => (key.toString, reprStr value)

structure EncoderState where
  -- All IDs and expression records are local to one theorem.
  -- Cache before flattening. Native Expr hashes are cached; structural equality
  -- also preserves binder names/annotations, unlike Expr's default alpha BEq.
  exprs : ExprStructMap Nat := HashMap.emptyWithCapacity 4096
  flatIds : HashMap FlatExpr Nat := HashMap.emptyWithCapacity 4096
  flatRecords : Array FlatExpr := #[]
  exprRecords : Array Json := #[]
  fvars : HashMap FVarId Nat := HashMap.emptyWithCapacity 256
  mvars : HashMap MVarId Nat := HashMap.emptyWithCapacity 256
  lmvars : HashMap LMVarId Nat := HashMap.emptyWithCapacity 32

abbrev EncodeM := StateT EncoderState (Except String)


/-- Assign a compact theorem-local ID on first use. -/
private def internFVar (id : FVarId) : EncodeM Nat :=
  modifyGet fun state =>
    let next := state.fvars.size
    let (existing?, fvars) := state.fvars.getThenInsertIfNew? id next
    (existing?.getD next, {state with fvars})

private def internMVar (id : MVarId) : EncodeM Nat :=
  modifyGet fun state =>
    let next := state.mvars.size
    let (existing?, mvars) := state.mvars.getThenInsertIfNew? id next
    (existing?.getD next, {state with mvars})

private def internLMVar (id : LMVarId) : EncodeM Nat :=
  modifyGet fun state =>
    let next := state.lmvars.size
    let (existing?, lmvars) := state.lmvars.getThenInsertIfNew? id next
    (existing?.getD next, {state with lmvars})

private def dumpLevel : Level → EncodeM Json
  | .zero => pure "zero"
  | .mvar id => return .mkObj [("mvar", ← internLMVar id)]
  | .succ inner => return .mkObj [("succ", ← dumpLevel inner)]
  | .max left right => return .mkObj [("max", .arr #[← dumpLevel left, ← dumpLevel right])]
  | .imax left right => return .mkObj [("imax", .arr #[← dumpLevel left, ← dumpLevel right])]
  | .param name => pure <| .mkObj [("param", name.toString)]

private def flatExprJson (expr : FlatExpr) : EncodeM Json := do
  match expr with
    | .bvar index => pure <| .mkObj [("bvar", index)]
    | .fvar id => pure <| .mkObj [("fvar", id)]
    | .mvar id => pure <| .mkObj [("mvar", id)]
    | .sort level => pure <| .mkObj [("sort", ← dumpLevel level)]
    | .const name levels => pure <| .mkObj [("const", .mkObj [
        ("name", name.toString),
        ("universes", (← levels.mapM dumpLevel).toJson)
      ])]
    | .app fn args => pure <| Json.mkObj [("app", Json.mkObj [
        ("fn", fn),
        ("args", args.toJson)
      ])]
    | .lam names type body binderInfo => pure <| .mkObj [("lambda", .mkObj [
        ("names", (names.map (·.toString)).toJson),
        ("type", type),
        ("body", body),
        ("binderInfo", binderInfoJson binderInfo)
      ])]
    | .forallE names type body binderInfo => pure <| .mkObj [("forall", .mkObj [
        ("names", (names.map (·.toString)).toJson),
        ("type", type),
        ("body", body),
        ("binderInfo", binderInfoJson binderInfo)
      ])]
    | .letE name type value body nondep => pure <| .mkObj [("let", .mkObj [
        ("name", name.toString),
        ("type", type),
        ("val", value),
        ("body", body),
        ("nondep", nondep)
      ])]
    | .lit (.natVal value) => pure <| .mkObj [("natural", toJson value)]
    | .lit (.strVal value) => pure <| .mkObj [("string", value)]
    | .mdata src inner =>
      let .mdata data _ := src.val | throw "metadata record lacks its source metadata"
      pure <| .mkObj [("metadata", .mkObj [
        ("data", kvMapJson data),
        ("expr", inner)
      ])]
    | .proj typeName index structExpr => pure <| .mkObj [("projection", .mkObj [
        ("typeName", typeName.toString),
        ("idx", index),
        ("struct", structExpr)
      ])]

/-- Deduplicate normalized records before allocating JSON or rendering names. -/
private def internFlatExpr (expr : FlatExpr) : EncodeM Nat := do
  if let some index := (← get).flatIds[expr]? then return index
  let record ← flatExprJson expr
  modifyGet fun state =>
    let index := state.exprRecords.size
    (index, {state with
      flatIds := state.flatIds.insert expr index
      flatRecords := state.flatRecords.push expr
      exprRecords := state.exprRecords.push record
    })

/-- Reflatten a function exposed by coercion erasure, preserving its arguments
and any trailing applications. Argument expressions are never spliced into it. -/
private def internApp (fn : Nat) (args : Array Nat) : EncodeM Nat := do
  if args.isEmpty then return fn
  let some head := (← get).flatRecords[fn]? | throw "application has an invalid function reference"
  match head with
  | .app inner initial => internFlatExpr (.app inner (initial ++ args))
  | _ => internFlatExpr (.app fn args)

/-- Cache raw structural expressions before any recursive preparation. Every
child takes this path too. IDs remain post-order even when a rewrite aliases a
wrapper to its value; normalized records share a second, shallow intern table. -/
private partial def dumpExpr (expr : Expr) : EncodeM Nat := do
  let key := ExprStructEq.mk expr
  if let some index := (← get).exprs[key]? then return index
  let index ← match expr with
    | .bvar idx => internFlatExpr (.bvar idx)
    | .fvar id => return ← internFlatExpr (.fvar (← internFVar id))
    | .mvar id => return ← internFlatExpr (.mvar (← internMVar id))
    | .sort lvl => internFlatExpr (.sort lvl)
    | .const name levels => internFlatExpr (.const name levels)
    | .app .. => expr.withApp fun fn args => do
      let (head, operands) ← match appPlan fn args with
        | .application fn args => do
          pure (← dumpExpr fn, ← args.mapM dumpExpr)
        | .operator family operands => do
          pure (← internFlatExpr (.const family []), ← operands.mapM dumpExpr)
      internApp head operands
    | .lam .. | .forallE .. => do
      let (names, type, body, info) := binderGroup expr
      let type ← dumpExpr type
      let body ← dumpExpr body
      internFlatExpr <| if expr.isLambda then .lam names type body info else .forallE names type body info
    | .letE name type value body nondep =>
      return ← internFlatExpr (.letE name (← dumpExpr type) (← dumpExpr value) (← dumpExpr body) nondep)
    | .lit literal => internFlatExpr (.lit literal)
    | .mdata _ inner => return ← internFlatExpr (.mdata key (← dumpExpr inner))
    | .proj typeName idx struct => return ← internFlatExpr (.proj typeName idx (← dumpExpr struct))
  modify fun state => {state with exprs := state.exprs.insert key index}
  return index

private def localInstanceSet (instances : LocalInstances) : HashSet FVarId :=
  instances.foldl (init := {}) fun acc inst => inst.fvar.fvarId?.elim acc acc.insert

private def dumpLocalDecl (instances : HashSet FVarId) (decl : LocalDecl) : EncodeM Json := do
  let common := [
    ("id", toJson (← internFVar decl.fvarId)),
    ("type", toJson (← dumpExpr decl.type)),
    ("kind", localDeclKindJson decl.kind),
    ("isInstance", toJson (instances.contains decl.fvarId))
  ]
  match decl with
  | .cdecl _ _ _ _ binderInfo _ =>
    return .mkObj <| common ++ [("binderInfo", binderInfoJson binderInfo)]
  | .ldecl _ _ _ _ value nondep _ =>
    return .mkObj <| common ++ [
      ("val", toJson (← dumpExpr value)),
      ("nondep", toJson nondep)
    ]

private def dumpLocalContext (decl : MetavarDecl) : EncodeM Json := do
  let instances := localInstanceSet decl.localInstances
  let declarations ← decl.lctx.foldlM (init := #[]) fun result localDecl => do
    return result.push (← dumpLocalDecl instances localDecl)
  return .arr declarations

structure ClosureState where
  -- Assignment traversal is state-dependent: never share these sets between
  -- tactic states, even when their immutable expression nodes are shared.
  exprSeen : ExprSet := {}
  levelSeen : HashSet Level := {}
  -- Each pair implements an insertion-ordered set: the hash set detects cycles,
  -- while the array fixes deterministic serialization order.
  mvarSeen : HashSet MVarId := {}
  mvars : Array MVarId := #[]
  lmvarSeen : HashSet LMVarId := {}
  lmvars : Array LMVarId := #[]

abbrev ClosureM := StateM ClosureState

private def markMVar (id : MVarId) : ClosureM Bool :=
  modifyGet fun state =>
    let (wasPresent, mvarSeen) := state.mvarSeen.containsThenInsert id
    if wasPresent then
      (false, state)
    else
      (true, {state with
        mvarSeen
        mvars := state.mvars.push id
      })

private def markLMVar (id : LMVarId) : ClosureM Bool :=
  modifyGet fun state =>
    let (wasPresent, lmvarSeen) := state.lmvarSeen.containsThenInsert id
    if wasPresent then
      (false, state)
    else
      (true, {state with
        lmvarSeen
        lmvars := state.lmvars.push id
      })

/- Find the transitive metavariable and universe-metavariable closure reachable
from the active target and local context. The seen sets both deduplicate records
and make cyclic or mutually-referential assignments terminate. -/
mutual
  private partial def collectLevel (mctx : MetavarContext) (level : Level) : ClosureM Unit := do
    unless level.hasMVar do return
    let seen ← modifyGet fun state =>
      let (seen, levelSeen) := state.levelSeen.containsThenInsert level
      (seen, {state with levelSeen})
    if seen then return
    match level with
    | .zero | .param _ => pure ()
    | .succ inner => collectLevel mctx inner
    | .max left right | .imax left right =>
      collectLevel mctx left
      collectLevel mctx right
    | .mvar id =>
      unless ← markLMVar id do return
      if let some assignment := mctx.lAssignment.find? id then
        collectLevel mctx assignment

  private partial def collectExpr (mctx : MetavarContext) (expr : Expr) : ClosureM Unit := do
    -- Lean caches this flag in each Expr, so irrelevant subtrees need no walk.
    unless expr.hasMVar do return
    let seen ← modifyGet fun state =>
      let (seen, exprSeen) := state.exprSeen.containsThenInsert expr
      (seen, {state with exprSeen})
    if seen then return
    match expr with
    | .bvar _ | .fvar _ | .lit _ => pure ()
    | .mvar id => collectMVar mctx id
    | .sort level => collectLevel mctx level
    | .const _ levels => levels.forM (collectLevel mctx)
    | .app fn arg =>
      collectExpr mctx fn
      collectExpr mctx arg
    | .lam _ type body _ | .forallE _ type body _ =>
      collectExpr mctx type
      collectExpr mctx body
    | .letE _ type value body _ =>
      collectExpr mctx type
      collectExpr mctx value
      collectExpr mctx body
    | .mdata _ inner => collectExpr mctx inner
    | .proj _ _ structExpr => collectExpr mctx structExpr

  private partial def collectLocalContext (mctx : MetavarContext)
      (lctx : LocalContext) : ClosureM Unit := do
    lctx.foldlM (init := ()) fun _ decl => do
      collectExpr mctx decl.type
      if let some value := decl.value? (allowNondep := true) then
        collectExpr mctx value

  private partial def collectMVar (mctx : MetavarContext) (id : MVarId) : ClosureM Unit := do
    unless ← markMVar id do return
    if let some decl := mctx.findDecl? id then
      collectExpr mctx decl.type
      collectLocalContext mctx decl.lctx
    if let some assignment := mctx.getExprAssignmentCore? id then
      collectExpr mctx assignment
    if let some delayed := mctx.getDelayedMVarAssignmentCore? id then
      delayed.fvars.forM (collectExpr mctx)
      collectMVar mctx delayed.mvarIdPending
end

private def collectStateClosure (mctx : MetavarContext) (goalDecl : MetavarDecl) : ClosureState :=
  let (_, closure) := (do
    collectExpr mctx goalDecl.type
    collectLocalContext mctx goalDecl.lctx).run {}
  closure

private def dumpNestedMVarDecl (decl : MetavarDecl) : EncodeM Json := do
  return .mkObj [
    ("type", ← dumpExpr decl.type),
    ("locals", ← dumpLocalContext decl),
    ("kind", metavarKindJson decl.kind)
  ]

private def dumpMVarState (mctx : MetavarContext) (id : MVarId) : EncodeM Json := do
  let assignment ← mctx.getExprAssignmentCore? id |>.mapM dumpExpr
  let delayed ← mctx.getDelayedMVarAssignmentCore? id |>.mapM fun value => do
    return Json.mkObj [
      ("fvars", toJson (← value.fvars.mapM dumpExpr)),
      ("pending", ← internMVar value.mvarIdPending)
    ]
  let declaration ← mctx.findDecl? id |>.mapM dumpNestedMVarDecl
  return .mkObj [
    ("id", ← internMVar id),
    ("decl", toJson declaration),
    ("assignment", toJson assignment),
    ("delayedAssign", toJson delayed)
  ]

private def dumpLMVarState (mctx : MetavarContext) (id : LMVarId) : EncodeM Json := do
  let declaration := match mctx.lDecls.find? id with
    | some decl => .mkObj [("depth", decl.depth), ("idx", decl.index)]
    | none => .null
  let assignment ← mctx.lAssignment.find? id |>.mapM dumpLevel
  return .mkObj [
    ("id", ← internLMVar id),
    ("decl", declaration),
    ("assignment", toJson assignment)
  ]

/-- Encode the active goal before a tactic. Other open goals are represented only
by `openGoalCount`; they are not part of the classifier's proof-state input. -/
private def dumpState (info : TacticInfo) : EncodeM Json := do
  let some goal := info.goalsBefore.head? | throw "transition has no active goal"
  let some goalDecl := info.mctxBefore.findDecl? goal | throw "active goal is absent from mctxBefore"
  let closure := collectStateClosure info.mctxBefore goalDecl
  let mvars ← closure.mvars.mapM (dumpMVarState info.mctxBefore)
  let lmvars ← closure.lmvars.mapM (dumpLMVarState info.mctxBefore)
  return .mkObj [
    ("target", ← dumpExpr goalDecl.type),
    ("locals", ← dumpLocalContext goalDecl),
    ("mvars", mvars.toJson),
    ("lvlMvars", lmvars.toJson)
  ]

private def sourceText (input : String) (source : SourceKey) : String :=
  (input.toRawSubstring.extract source.start source.stop).toString

private def dumpTrn (input : String) (transition : RawTrn) : EncodeM Json := do
  let {info, source} := transition
  return .mkObj [
    ("tactic", .mkObj [
      ("kind", source.kind.toString),
      ("src", sourceText input source)
    ]),
    ("srcSpan", .arr #[source.start.byteIdx, source.stop.byteIdx]),
    ("openGoalCount", info.goalsBefore.length),
    ("state", ← dumpState info)
  ]

private def declarationSpan (fileMap : FileMap) (ranges : Option DeclarationRanges) : Json :=
  match ranges with
  | some ranges =>
    let start := (fileMap.ofPosition ranges.range.pos).byteIdx
    let stop := (fileMap.ofPosition ranges.range.endPos).byteIdx
    .arr #[start, stop]
  | none => .null

/-- Build one self-contained theorem record. Expression and variable IDs are
shared by its transitions, but reset before the next theorem. -/
private def dumpTheorem (input : String) (fileMap : FileMap) (env : Environment)
    (moduleName declaration : Name) (trns : Array RawTrn) : IO Json := do
  let (ranges, _, _) ← Lean.Meta.MetaM.toIO (findDeclarationRanges? declaration)
    {fileName := moduleName.toString, fileMap} {env}
  let (trnRecords, encoder) ←
    match (trns.mapM (dumpTrn input)).run {} with
    | .ok result => pure result
    | .error error => throw <| IO.userError s!"{declaration}: {error}"
  return .mkObj [
    ("name", declaration.toString),
    ("module", moduleName.toString),
    ("srcSpan", declarationSpan fileMap ranges),
    ("exprs", encoder.exprRecords.toJson),
    ("trns", trnRecords.toJson)
  ]

structure EmissionTiming where
  -- Building the Json also traverses and interns elaborated expressions.
  jsonBuild : Nat := 0
  jsonEncode : Nat := 0

private def emitTheorems (input : String) (moduleName : Name) (env : Environment)
    (collection : CollectionState) (showTiming : Bool) : IO EmissionTiming := do
  let fileMap := FileMap.ofString input
  let mut timing : EmissionTiming := {}
  for declaration in collection.order do
    let some constantInfo := env.find? declaration | continue
    unless constantInfo.isTheorem do continue
    let some trns := collection.trns[declaration]? | continue
    let started ← if showTiming then IO.monoMsNow else pure 0
    let record ← dumpTheorem input fileMap env moduleName declaration trns
    let builtAt ← if showTiming then IO.monoMsNow else pure 0
    let encoded := record.compress
    let encodedAt ← if showTiming then IO.monoMsNow else pure 0
    IO.println encoded
    if showTiming then
      timing := {
        jsonBuild := timing.jsonBuild + builtAt - started
        jsonEncode := timing.jsonEncode + encodedAt - builtAt
      }
  return timing

/-! ## File elaboration -/

structure FileTiming where
  imports : Nat
  elaboration : Nat
  selection : Nat
  jsonBuild : Nat
  jsonEncode : Nat
  total : Nat

private def reportTiming (timing : FileTiming) : IO Unit :=
  IO.eprintln <| s!"TIMING imports={timing.imports}ms elaboration={timing.elaboration}ms " ++
    s!"selection={timing.selection}ms json_build={timing.jsonBuild}ms " ++
    s!"json_encode={timing.jsonEncode}ms total={timing.total}ms"

private def abortOnErrors (path : FilePath) (phase : String) (messages : MessageLog) : IO Unit := do
  unless messages.hasErrors do return
  for message in messages.toList do
    if message.severity == .error then
      IO.eprintln s!"ERROR: {← message.toString}"
  throw <| IO.userError s!"{path}: errors during {phase}; aborting"

/-- Re-elaborate one source module with information trees enabled, then emit its
selected theorem records. Selection happens before any expression traversal or
source-string extraction, keeping discarded elaborator nodes cheap. -/
unsafe def processFile (target : ExtractionTarget) (showTiming : Bool) : IO Unit := do
  let {path, moduleName, options} := target
  -- With async disabled, `backwards.privateInPublic` crashes the elaborator.
  let options := Elab.async.setIfNotSet options true
  let totalStart ← IO.monoMsNow
  let input ← IO.FS.readFile path
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let parsedAt ← IO.monoMsNow
  -- Lean consumes this one-shot permission when an import environment is loaded.
  -- It must therefore be renewed for each independently elaborated source file.
  enableInitializersExecution
  let (env, messages) ← processHeader header options messages inputCtx
  let importedAt ← IO.monoMsNow
  abortOnErrors path "import" messages

  let env := env.setMainModule moduleName
  let commandState := {Command.mkState env messages options with infoState.enabled := true}
  let finalState := (← IO.processCommands inputCtx parserState commandState).commandState
  let elaboratedAt ← IO.monoMsNow
  abortOnErrors path "elaboration" finalState.messages

  let trees := finalState.infoState.trees.toArray
  let sources := buildSourceIndex trees
  let collection ← collectTransitions sources trees
  let selectedAt ← IO.monoMsNow
  let emissionTiming ← emitTheorems input moduleName finalState.env collection showTiming
  let finishedAt ← IO.monoMsNow
  if showTiming then
    reportTiming {
      imports := importedAt - parsedAt
      elaboration := elaboratedAt - importedAt
      selection := selectedAt - elaboratedAt
      jsonBuild := emissionTiming.jsonBuild
      jsonEncode := emissionTiming.jsonEncode
      total := finishedAt - totalStart
    }

/-! ## Command-line interface -/

structure CliConfig where
  timing : Bool := false
  help : Bool := false
  path? : Option FilePath := none

private def usage := String.intercalate "\n" [
  "Usage: trustmebro-extract-state [--timing] FILE",
  "",
  "Extract one Lean source file, writing one theorem object per line to stdout.",
  "The Python supervisor handles file lists and concurrent workers.",
  "",
  "  --timing           Report per-file phase timings to stderr",
  "  -h, --help         Show this help"
]

private def parseArgs (args : List String) : Except String CliConfig :=
  let rec loop (args : List String) (config : CliConfig) : Except String CliConfig := do
    match args with
    | [] => return config
    | ["--", path] => return {config with path? := some (FilePath.mk path)}
    | "--timing" :: rest => loop rest {config with timing := true}
    | "-h" :: rest => loop rest {config with help := true}
    | "--help" :: rest => loop rest {config with help := true}
    | arg :: rest =>
      if arg.startsWith "-" then
        throw s!"unknown option: {arg}"
      if config.path?.isSome then throw "expected exactly one source file"
      loop rest {config with path? := some (FilePath.mk arg)}
  loop args {}

end Trustmebro.Extraction

open Trustmebro.Extraction

unsafe def main (args : List String) : IO Unit := do
  let config ← match parseArgs args with
    | .ok config => pure config
    | .error error => throw <| IO.userError s!"{error}\n\n{usage}"
  if config.help then
    IO.println usage
    return
  let some path := config.path?
    | throw <| IO.userError s!"expected one source file\n\n{usage}"

  initSearchPath (← findSysroot)
  let workspaceStart ← IO.monoMsNow
  let workspace ← loadWorkspace
  let target ← resolveExtractionTarget workspace path
  let workspaceFinished ← IO.monoMsNow
  if config.timing then
    IO.eprintln s!"TIMING setup: {workspaceFinished - workspaceStart}ms"
  processFile target config.timing
