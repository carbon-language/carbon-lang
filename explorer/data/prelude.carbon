// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Carbon api;

// ----------------------
// Conversion interfaces.
// ----------------------

// Explicitly convert `Self` to `T`.
interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

// Implicitly convert `Self` to `T`.
interface ImplicitAs(T:! type) {
  extends As(T);
}

// TODO: This should be private.
interface __EqualConverter {
  let T:! type;
  fn Convert(t: T) -> Self;
}
fn __EqualConvert[T:! type](t: T, U:! __EqualConverter where .T = T) -> U {
  return U.Convert(t);
}
impl forall [U:! type] U as __EqualConverter where .T = U {
  fn Convert(u: U) -> U { return u; }
}

// Every type implicitly converts to single-step-equal types.
impl forall [T:! type, U:! type where .Self == T] T as ImplicitAs(U) {
  fn Convert[self: Self]() -> U { return __EqualConvert(self, U); }
}

// TODO: Simplify this once we have variadics.
// TODO: Should these be final?
impl forall [U1:! type, T1:! ImplicitAs(U1)]
    (T1,) as ImplicitAs((U1,)) {
  fn Convert[self: Self]() -> (U1,) {
    let (v1: T1,) = self;
    return (v1.Convert(),);
  }
}
impl forall [U1:! type, U2:! type, T1:! ImplicitAs(U1), T2:! ImplicitAs(U2)]
    (T1, T2) as ImplicitAs((U1, U2)) {
  fn Convert[self: Self]() -> (U1, U2) {
    let (v1: T1, v2: T2) = self;
    return (v1.Convert(), v2.Convert());
  }
}
impl forall [U1:! type, U2:! type, U3:! type,
             T1:! ImplicitAs(U1), T2:! ImplicitAs(U2), T3:! ImplicitAs(U3)]
    (T1, T2, T3) as ImplicitAs((U1, U2, U3)) {
  fn Convert[self: Self]() -> (U1, U2, U3) {
    let (v1: T1, v2: T2, v3: T3) = self;
    return (v1.Convert(), v2.Convert(), v3.Convert());
  }
}

// ----------------------
// Comparison interfaces.
// ----------------------

// ----------------------
// EQUAL
// ----------------------

interface EqWith(U:! type) {
  fn Equal[self: Self](other: U) -> bool;
  fn NotEqual[self: Self](other: U) -> bool;
}

constraint Eq {
  extends EqWith(Self);
}

// TODO: Simplify this once we have variadics
impl forall [T2:! type, U2:! type, T1:! EqWith(T2), U1:! EqWith(U2)]
    (T1, U1) as EqWith((T2, U2)) {
  fn Equal[self: Self](other: (T2, U2)) -> bool {
    let (l1: T1, l2: U1) = self;
    let (r1: T2, r2: U2) = other;
    return l1 == r1 and l2 == r2;
  }
  fn NotEqual[self: Self](other: (T2, U2)) -> bool {
      let (l1: T1, l2: U1) = self;
      let (r1: T2, r2: U2) = other;
      return l1 != r1 or l2 != r2;
    }
}

impl bool as EqWith(Self) {
  fn Equal[self: Self](other: Self) -> bool {
    return if self then other else not other;
  }
  fn NotEqual[self: Self](other: Self) -> bool {
    return if self then not other else other;
  }
}

impl i32 as EqWith(Self) {
  fn Equal[self: Self](other: Self) -> bool {
    return __intrinsic_int_eq(self, other);
  }

  fn NotEqual[self: Self](other: Self) -> bool {
    return not __intrinsic_int_eq(self, other);
  }
}

impl String as EqWith(Self) {
  fn Equal[self: Self](other: Self) -> bool {
    return __intrinsic_str_eq(self, other);
  }

  fn NotEqual[self: Self](other: Self) -> bool {
    return not __intrinsic_str_eq(self, other);
  }
}

// ----------------------
// COMPARE
// ----------------------

choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}

// TODO: Per the design, this should be named `OrderedWith`.
interface CompareWith(U:! type) {
  fn Compare[self: Self](u: U) -> Ordering;
  // TODO: Add `default fn` for Less, LessOrEquivalent, Greater, and GreaterOrEquivalent once it's available.
}
constraint Ordered {
  extends CompareWith(Self);
}

impl i32 as CompareWith(Self) {
  fn Compare[self: Self](other: Self) -> Ordering {
    var comp: i32 = __intrinsic_int_compare(self, other);
    if (comp == -1) {
      return Ordering.Less();
    }
    if (comp == 0) {
      return Ordering.Equivalent();
    }
    if (comp == 1) {
      return Ordering.Greater();
    }
    return Ordering.Incomparable();

  }
}

impl String as CompareWith(Self) {
  fn Compare[self: Self](other: Self) -> Ordering {
    var comp: i32 = __intrinsic_str_compare(self, other);
    if (comp == -1) {
      return Ordering.Less();
    }
    if (comp == 0) {
      return Ordering.Equivalent();
    }
    if (comp == 1) {
      return Ordering.Greater();
    }
    return Ordering.Incomparable();
  }
}

interface LessWith(U:! type) {
  fn Less[self: Self](other: U) -> bool;
}

interface LessEqWith(U:! type) {
  fn LessEq[self: Self](other: U) -> bool;
}

interface GreaterWith(U:! type) {
  fn Greater[self: Self](other: U) -> bool;
}

interface GreaterEqWith(U:! type) {
  fn GreaterEq[self: Self](other: U) -> bool;
}

impl i32 as LessWith(Self) {
  fn Less[self: Self](other: Self) -> bool {
    var comp: Ordering = self.(CompareWith(i32).Compare)(other);
    match (comp) {
      case Ordering.Less() => {
        return true;
      }
    }
    return false;
  }
}

impl String as LessWith(Self) {
  fn Less[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Less() => {
        return true;
      }
    }
    return false;
  }
}

impl i32 as LessEqWith(Self) {
  fn LessEq[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(i32).Compare)(other);
    match(comp){
      case Ordering.Less() => {
        return true;
      }
      case Ordering.Equivalent() => {
        return true;
      }
    }
    return false;
  }
}

impl String as LessEqWith(Self) {
  fn LessEq[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Less() => {
        return true;
      }
      case Ordering.Equivalent() => {
        return true;
      }
    }
    return false;
  }
}

impl i32 as GreaterWith(Self) {
  fn Greater[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(i32).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
    }
    return false;
  }
}

impl String as GreaterWith(Self) {
  fn Greater[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
    }
    return false;
  }
}

impl i32 as GreaterEqWith(Self) {
  fn GreaterEq[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(i32).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
      case Ordering.Equivalent() => {
        return true;
      }
    }
    return false;
  }
}

impl String as GreaterEqWith(Self) {
  fn GreaterEq[self: Self](other: Self) -> bool {
    var comp: Ordering =  self.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
      case Ordering.Equivalent() => {
        return true;
      }
    }
    return false;
  }
}

// ----------------------
// Arithmetic interfaces.
// ----------------------

interface Negate {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

interface AddWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint Add {
  extends AddWith(Self) where .Result = Self;
}

interface SubWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint Sub {
  extends SubWith(Self) where .Result = Self;
}

interface MulWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint Mul {
  extends MulWith(Self) where .Result = Self;
}

interface DivWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint Div {
  extends DivWith(Self) where .Result = Self;
}

interface ModWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint Mod {
  extends ModWith(Self) where .Result = Self;
}

// Note, these impls use the builtin addition for i32.
external impl i32 as Negate where .Result = i32 {
  fn Op[self: i32]() -> i32 { return -self; }
}
external impl i32 as AddWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 { return self + other; }
}
external impl i32 as SubWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 { return self - other; }
}
external impl i32 as MulWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 { return self * other; }
}
external impl i32 as DivWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 { return self / other; }
}
external impl i32 as ModWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 { return self % other; }
}

// ---------------------------------
// Bitwise and bit-shift interfaces.
// ---------------------------------

// Unary `^`.
interface BitComplement {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

// Binary `&`.
interface BitAndWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint BitAnd {
  extends BitAndWith(Self) where .Result = Self;
}

// Binary `|`.
interface BitOrWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint BitOr {
  extends BitOrWith(Self) where .Result = Self;
}

// Binary `^`.
interface BitXorWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint BitXor {
  extends BitXorWith(Self) where .Result = Self;
}

// Binary `<<`.
interface LeftShiftWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint LeftShift {
  extends LeftShiftWith(Self) where .Result = Self;
}

// Binary `>>`.
interface RightShiftWith(U:! type) {
  // TODO: = Self
  let Result:! type;
  fn Op[self: Self](other: U) -> Result;
}
constraint RightShift {
  extends RightShiftWith(Self) where .Result = Self;
}

external impl i32 as BitComplement where .Result = i32 {
  fn Op[self: i32]() -> i32 {
    return __intrinsic_int_bit_complement(self);
  }
}
external impl i32 as BitAndWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_and(self, other);
  }
}
external impl i32 as BitOrWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_or(self, other);
  }
}
external impl i32 as BitXorWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_xor(self, other);
  }
}
external impl i32 as LeftShiftWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 {
    return __intrinsic_int_left_shift(self, other);
  }
}
external impl i32 as RightShiftWith(i32) where .Result = i32 {
  fn Op[self: i32](other: i32) -> i32 {
    return __intrinsic_int_right_shift(self, other);
  }
}

//-------------------------
// Optional
//-------------------------
choice OptionalElement(T:! type) {
  None(),
  Element(T)
}

class Optional(T:! type) {
  fn CreateEmpty() -> Optional(T) {
    return {.element = OptionalElement(T).None()};
  }
  fn Create(value: T) -> Optional(T) {
    return {.element = OptionalElement(T).Element(value)};
  }

  fn HasValue[self: Self]() -> bool {
    match(self.element) {
      case OptionalElement(T).None() => { return false; }
    }
    return true;
  }

  fn Get[self: Self]() -> T {
    var y: T;
    match(self.element) {
      case OptionalElement(T).Element(x: T) => {
        return x;
      }
    }
    // TODO: Use assert as soon as available
    return y;
  }

  var element: OptionalElement(T);
}

// ------------------------
// Miscellaneous utilities.
// ------------------------

// Note that Print is experimental, and not part of an accepted proposal, but
// is included here for printing state in tests.
// TODO: Remove Print special casing once we have variadics or overloads.
// fn Print(format_str: String) {
//   __intrinsic_print(format_str);
// }

fn Assert(condition: bool, message: String){
    __intrinsic_assert(condition, message);
}

fn Rand(low: i32, high: i32) -> i32{
    return __intrinsic_rand(low,high);
}

class Heap {
  fn New[T:! type, self: Self](x : T) -> T* {
    return __intrinsic_new(x);
  }
  fn Delete[T:! type, self: Self](p : T*) {
    __intrinsic_delete(p);
  }
}

var heap: Heap = {};
