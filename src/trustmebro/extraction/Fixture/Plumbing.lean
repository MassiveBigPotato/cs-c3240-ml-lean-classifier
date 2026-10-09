import Mathlib.Data.Set.Basic
import Mathlib.Algebra.Group.Pi.Basic
import Mathlib.Algebra.Group.Hom.Defs
import Mathlib.Algebra.Group.Nat.Defs

namespace PlumbingProbe

theorem membership (s : Set Nat) (x : Nat) : x ∈ s ↔ x ∈ s := by rfl

theorem arithmetic (x : Nat) : x + 2 = x + 2 := by rfl

theorem functionArithmetic (f g : Nat → Nat) (x : Nat) : (f + g) x = (f + g) x := by rfl

theorem partialOperator
    (_h : @HAdd.hAdd Nat Nat Nat = @HAdd.hAdd Nat Nat Nat) : True := by trivial

theorem subtype (p : Nat → Prop) (x : Subtype p) : x.val = x.val := by rfl

theorem hom (f : Nat →+ Nat) (x : Nat) : f x = f x := by rfl

theorem composedHom (f g : Nat →+ Nat) (x : Nat) : (f.comp g) x = (f.comp g) x := by rfl

theorem cast (n : Nat) : (n : Int) = (n : Int) := by rfl

theorem genericCast [Coe Nat Int] (n : Nat) : (Coe.coe n : Int) = Coe.coe n := by rfl

theorem typeCoercion [CoeSort Nat (Type)] (n : Nat) (x : (CoeSort.coe n : Type)) : x = x := by rfl

end PlumbingProbe
