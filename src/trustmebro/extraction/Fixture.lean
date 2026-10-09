import Lean.Elab.Tactic.Omega
import Mathlib.Tactic.FailIfNoProgress
import Mathlib.Tactic.Ring

/-!
Regression corpus for the proof-state extractor.

The examples exercise source tactics, structural wrappers, rollback behavior,
macro expansion, and representative elaborated expression forms. They are not
intended as training data.
-/

namespace Trustmebro.Extraction.Fixture

set_option linter.unusedTactic false

syntax (name := fixtureExact) "fixture_exact " term : tactic
macro_rules
  | `(tactic| fixture_exact $term) => `(tactic| exact $term)

syntax (name := fixtureMacroPartialFailure) "fixture_macro_partial_failure" : tactic
macro_rules
  | `(tactic| fixture_macro_partial_failure) => `(tactic| constructor; fail)

elab "fixture_elab_assumption" : tactic => do
  Lean.Elab.Tactic.evalTactic (← `(tactic| assumption))

elab "fixture_elab_failure" : tactic => do
  throwError "intentional fixture failure"

elab "fixture_elab_partial_failure" : tactic => do
  Lean.Elab.Tactic.evalTactic (← `(tactic| constructor))
  throwError "intentional fixture failure after progress"

theorem direct (p : Prop) (hp : p) : p := by
  exact hp

theorem sourceMacro (p : Prop) (hp : p) : p := by
  fixture_exact hp

theorem anonymousConstructor (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  exact ⟨hp, hq⟩

theorem bullets (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  · exact hp
  · exact hq

theorem namedCases
    (p q r : Prop) (h : p ∨ q) (hp : p → r) (hq : q → r) : r := by
  cases h with
  | inl h => exact hp h
  | inr h => exact hq h

theorem namedCasePrime (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  case' left => exact hp
  case' right => exact hq

theorem nextCases (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  next => exact hp
  next => exact hq

theorem sequencing (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor <;> assumption

theorem semicolonSequencing (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor; assumption
  assumption

theorem nestedCombinators (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor <;> first | exact hp | exact hq

theorem parenthesized (p : Prop) (hp : p) : p := by
  (exact hp)

theorem focused (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  focus exact hp
  exact hq

theorem allGoals (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  all_goals assumption

theorem firstBranch (p : Prop) (hp : p) : p := by
  first
  | rfl
  | exact hp

theorem firstMacro (p : Prop) (hp : p) : p := by
  first
  | fail
  | fixture_exact hp

theorem firstElaborator (p : Prop) (hp : p) : p := by
  first
  | fixture_elab_failure
  | fixture_elab_assumption

theorem firstRing (x y : ℤ) : (x + y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 := by
  first
  | fail
  | ring

theorem firstOmega (m n : ℕ) (h : m ≤ n) : m ≤ n + 1 := by
  first
  | fail
  | omega

theorem firstAutomationFallback (p : Prop) (hp : p) : p := by
  first
  | ring
  | fixture_exact hp

theorem solveBranch (p : Prop) (hp : p) : p := by
  solve
  | rfl
  | exact hp

theorem trySuccess (p : Prop) (hp : p) : p := by
  try exact hp

theorem tryFailure (p : Prop) (hp : p) : p := by
  try rfl
  exact hp

theorem tryMacroRollback (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  try fixture_macro_partial_failure
  exact ⟨hp, hq⟩

theorem tryElaboratorRollback (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  try fixture_elab_partial_failure
  exact ⟨hp, hq⟩

theorem solveRollback (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  solve
  | fixture_macro_partial_failure
  | exact ⟨hp, hq⟩

theorem progressGuard (p : Prop) (hp : p) : p := by
  fail_if_no_progress exact hp

theorem repeatFailure (p : Prop) (hp : p) : p := by
  repeat rfl
  exact hp

theorem repeatPrimeFailure (p : Prop) (hp : p) : p := by
  repeat' rfl
  exact hp

theorem repeatMacroRollback (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  repeat fixture_macro_partial_failure
  exact ⟨hp, hq⟩

theorem repeatElaboratorRollback (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  repeat fixture_elab_partial_failure
  exact ⟨hp, hq⟩

theorem repeatPrimeSuccess (p : Prop) (hp : p) : (p ∧ p) ∧ (p ∧ p) := by
  repeat' constructor
  all_goals exact hp

theorem repeatOneOrMore (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  repeat1' constructor
  all_goals assumption

theorem anyGoalsPartial (p : Prop) (hp : p) : True ∧ p := by
  constructor
  any_goals constructor
  exact hp

theorem anyGoalsRollback (p q : Prop) (hp : p) (hq : q) : True ∧ (p ∧ q) := by
  constructor
  any_goals (first | exact True.intro | (constructor; fail))
  exact ⟨hp, hq⟩

theorem anyGoalsMacroRollback (p q : Prop) (hp : p) (hq : q) : True ∧ (p ∧ q) := by
  constructor
  any_goals (first | exact True.intro | fixture_macro_partial_failure)
  exact ⟨hp, hq⟩

theorem anyGoalsElaboratorRollback (p q : Prop) (hp : p) (hq : q) : True ∧ (p ∧ q) := by
  constructor
  any_goals (first | exact True.intro | fixture_elab_partial_failure)
  exact ⟨hp, hq⟩

theorem anyGoalsAutomationRollback (x : ℤ) : True ∧ (x + 0 = x) := by
  constructor
  any_goals (first | exact True.intro | (ring_nf; fail))
  ring

theorem localDeclarations (n : Nat) : n = n := by
  let copy := n
  have copied : copy = n := rfl
  exact copied.symm.trans copied

theorem localInstance {α : Type} [Inhabited α] : Nonempty α := by
  exact ⟨default⟩

universe u v

theorem universesAndDependentBinders
    {α : Type u} {β : α → Type v} (f : (x : α) → β x) (x : α) : f x = f x := by
  rfl

theorem naturalLiteral : (184467440737095516160 : Nat) = 184467440737095516160 := by
  rfl

theorem stringLiteral : "fixture" = "fixture" := by
  rfl

structure Box (α : Type u) where
  value : α

theorem projection {α : Type u} (box : Box α) : box.value = box.value := by
  rfl

theorem ringExample (x y : ℤ) : (x + y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 := by
  ring

theorem omegaExample (m n : ℕ) (h : m ≤ n) : m ≤ n + 1 := by
  omega

/- Cache regression: a tiny shared DAG has billions of expanded tree nodes.
Converting it recursively before checking the cache would explode. -/
elab "fixture_shared_prop" : term => do
  let mut expr := Lean.mkNatLit 0
  for _ in [:32] do
    expr := Lean.mkApp2 (Lean.mkConst ``Nat.add) expr expr
  Lean.Meta.mkEq expr expr

theorem sharedExpressionDag : fixture_shared_prop := by
  rfl

def BinderDomain (_ : Nat) := Nat

-- Equal raw bvar indices in successive domains refer to different binders.
theorem differentlyScopedDomains (a : Nat) : ∀ (b : BinderDomain a) (c : BinderDomain b), True := by
  intro b c
  trivial

-- The shifted domains here really are the same outer α, so grouping is safe.
theorem sharedOuterDomain (α : Type) : ∀ (x y : α), True := by
  intro x y
  trivial

theorem sharedLambdaDomain (α : Type) : (fun (x y : α) => x) = (fun (x y : α) => x) := by
  rfl

theorem distinctBinderAnnotations : (fun (left : Nat) => left) = (fun (right : Nat) => right) := by
  rfl

elab "fixture_wide_prop" : term => do
  let mut fn := Lean.mkNatLit 0
  for idx in [:128] do
    fn := Lean.mkLambda (.str .anonymous s!"x{idx}") .default (Lean.mkConst ``Nat) fn
  let expr := Lean.mkAppN fn (Array.replicate 128 (Lean.mkNatLit 0))
  Lean.Meta.mkEq expr expr

theorem wideApplication : fixture_wide_prop := by
  rfl

/- Representation rewrite regressions: the kernel checks the original terms;
only the exported feature graph is normalized. -/
theorem normalizedNatAdd (x y : Nat) : Nat.add x y = Nat.add x y := by
  rfl

theorem normalizedIntAdd (x y : Int) : Int.add x y = Int.add x y := by
  rfl

theorem normalizedOverloadedAdd (x y : Nat) : x + y = x + y := by
  rfl

theorem normalizedNatArithmetic (x y : Nat) :
    Nat.add x y = Nat.add x y ∧ Nat.mul x y = Nat.mul x y ∧
    Nat.sub x y = Nat.sub x y ∧ Nat.div x y = Nat.div x y := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem normalizedFinArithmetic (n : Nat) (x y : Fin (n + 1)) :
    Fin.add x y = Fin.add x y ∧ Fin.mul x y = Fin.mul x y ∧
    Fin.sub x y = Fin.sub x y ∧ Fin.div x y = Fin.div x y := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem bareOperator : @HAdd.hAdd.{0, 0, 0} = @HAdd.hAdd.{0, 0, 0} := by
  rfl

theorem partialOperator : @HAdd.hAdd Nat Nat = @HAdd.hAdd Nat Nat := by
  rfl

theorem partialNativeOperator (x : Nat) : Nat.add x = Nat.add x := by
  rfl

theorem normalizedNatCast (n : Nat) : Nat.cast (R := Int) n = Nat.cast (R := Int) n := by
  rfl

structure CoercionBox where
  val : Nat

instance : Coe CoercionBox Nat := ⟨CoercionBox.val⟩

theorem partialCoercion : @Coe.coe CoercionBox Nat = @Coe.coe CoercionBox Nat := by
  rfl

theorem normalizedCoercion (x : CoercionBox) :
    Coe.coe (α := CoercionBox) (β := Nat) x = Coe.coe (α := CoercionBox) (β := Nat) x := by
  rfl

instance : CoeDep Nat 7 Bool := ⟨true⟩

theorem normalizedDependentCoercion :
    @CoeDep.coe Nat 7 Bool inferInstance = @CoeDep.coe Nat 7 Bool inferInstance := by
  rfl

structure FunctionBox where
  fn : Nat → Nat

instance : CoeFun FunctionBox (fun _ => Nat → Nat) := ⟨FunctionBox.fn⟩

def makeFunctionBox (n : Nat) : FunctionBox := ⟨fun x => n + x⟩

theorem normalizedFunctionCoercion (box : FunctionBox) (x : Nat) : box x = box x := by
  rfl

theorem normalizedAppliedFunctionCoercion (n x : Nat) :
    (makeFunctionBox n) x = (makeFunctionBox n) x := by
  rfl

-- Ordinary elaboration sometimes unfolds CoeFun.coe to the actual projection.
-- Build the unexpanded application too, to exercise its trailing operands.
elab "fixture_raw_function_coercion " applied:num : term => do
  let box ← if applied.getNat == 0 then
    Lean.Meta.mkAppM ``FunctionBox.mk
      #[Lean.mkLambda `x .default (Lean.mkConst ``Nat) (Lean.mkBVar 0)]
    else Lean.Meta.mkAppM ``makeFunctionBox #[Lean.mkNatLit 2]
  let fn ← Lean.Meta.mkAppM ``CoeFun.coe #[box]
  let expr := Lean.mkApp fn (Lean.mkNatLit 3)
  Lean.Meta.mkEq expr expr

theorem rawFunctionCoercion : fixture_raw_function_coercion 0 := by
  rfl

theorem rawAppliedFunctionCoercion : fixture_raw_function_coercion 1 := by
  rfl

theorem nestedApplications (f g : Nat → Nat → Nat → Nat → Nat) (a b c x y z w : Nat) :
    f a (g x y z w) b c = f a (g x y z w) b c := by
  rfl

elab "fixture_binder_info_prop" : term => do
  let domain := Lean.mkConst ``Nat
  let explicit := Lean.mkLambda `x .default domain (Lean.mkBVar 0)
  let implicit := Lean.mkLambda `x .implicit domain (Lean.mkBVar 0)
  Lean.Meta.mkEq explicit implicit

theorem distinctBinderKinds : fixture_binder_info_prop := by
  rfl

end Trustmebro.Extraction.Fixture
