// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Carbon api;

// ----------------------
// Conversion interfaces.
// ----------------------

// Explicitly convert `Self` to `T`.
interface As(T:! Type) {
  fn Convert[me: Self]() -> T;
}

// Implicitly convert `Self` to `T`.
interface ImplicitAs(T:! Type) {
  fn Convert[me: Self]() -> T;
}

// TODO: ImplicitAs(T) should extend As(T).
impl forall [T:! Type, U:! ImplicitAs(T)] U as As(T) {
  fn Convert[me: Self]() -> T { return me.Convert(); }
}

// Every type implicitly converts to itself.
impl forall [T:! Type] T as ImplicitAs(T) {
  fn Convert[me: Self]() -> T { return me; }
}

// TODO: Simplify this once we have variadics.
// TODO: Should these be final?
impl forall [U1:! Type, T1:! ImplicitAs(U1)]
    (T1,) as ImplicitAs((U1,)) {
  fn Convert[me: Self]() -> (U1,) {
    let (v1: T1,) = me;
    return (v1.Convert(),);
  }
}
impl forall [U1:! Type, U2:! Type, T1:! ImplicitAs(U1), T2:! ImplicitAs(U2)]
    (T1, T2) as ImplicitAs((U1, U2)) {
  fn Convert[me: Self]() -> (U1, U2) {
    let (v1: T1, v2: T2) = me;
    return (v1.Convert(), v2.Convert());
  }
}
impl forall [U1:! Type, U2:! Type, U3:! Type,
             T1:! ImplicitAs(U1), T2:! ImplicitAs(U2), T3:! ImplicitAs(U3)]
    (T1, T2, T3) as ImplicitAs((U1, U2, U3)) {
  fn Convert[me: Self]() -> (U1, U2, U3) {
    let (v1: T1, v2: T2, v3: T3) = me;
    return (v1.Convert(), v2.Convert(), v3.Convert());
  }
}

// ----------------------
// Comparison interfaces.
// ----------------------

// ----------------------
// EQUAL
// ----------------------

interface EqWith(U:! Type) {
  fn Equal[me: Self](other: U) -> bool;
  // TODO: NotEqual with default impl
}
// TODO: constraint Eq { ... }


// TODO: Simplify this once we have variadics
impl forall [T2:! Type, U2:! Type, T1:! EqWith(T2), U1:! EqWith(U2)]
    (T1, U1) as EqWith((T2, U2)) {
  fn Equal[me: Self](other: (T2, U2)) -> bool {
    let (l1: T1, l2: U1) = me;
    let (r1: T2, r2: U2) = other;
    return l1 == r1 and l2 == r2;
  }
}

impl i32 as EqWith(Self) {
  fn Equal[me: Self](other: Self) -> bool {
    return __intrinsic_int_eq(me, other);
  }
}

impl String as EqWith(Self) {
  fn Equal[me: Self](other: Self) -> bool {
    return __intrinsic_str_eq(me, other);
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

interface CompareWith(U:! Type) {
  fn Compare[me: Self](u: U) -> Ordering;
  // TODO: Add `default fn` for Less, LessOrEquivalent, Greater, and GreaterOrEquivalent once it's available.
}
// TODO: constraint Ordered { ... }

impl i32 as CompareWith(Self) {
  fn Compare[me: Self](other: Self) -> Ordering {
    var comp: i32 = __intrinsic_int_compare(me, other);
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
  fn Compare[me: Self](other: Self) -> Ordering {
    var comp: i32 = __intrinsic_str_compare(me, other);
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

interface LessWith(U:! Type) {
  fn Less[me: Self](other: U) -> bool;
}

interface LessEqWith(U:! Type) {
  fn LessEq[me: Self](other: U) -> bool;
}

interface GreaterWith(U:! Type) {
  fn Greater[me: Self](other: U) -> bool;
}

interface GreaterEqWith(U:! Type) {
  fn GreaterEq[me: Self](other: U) -> bool;
}

impl i32 as LessWith(Self) {
  fn Less[me: Self](other: Self) -> bool {
    var comp: Ordering = me.(CompareWith(i32).Compare)(other);
    match (comp) {
      case Ordering.Less() => {
        return true;
      }
    }
    return false;
  }
}

impl String as LessWith(Self) {
  fn Less[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Less() => {
        return true;
      }
    }
    return false;
  }
}

impl i32 as LessEqWith(Self) {
  fn LessEq[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(i32).Compare)(other);
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
  fn LessEq[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(String).Compare)(other);
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
  fn Greater[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(i32).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
    }
    return false;
  }
}

impl String as GreaterWith(Self) {
  fn Greater[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(String).Compare)(other);
    match(comp){
      case Ordering.Greater() => {
        return true;
      }
    }
    return false;
  }
}

impl i32 as GreaterEqWith(Self) {
  fn GreaterEq[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(i32).Compare)(other);
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
  fn GreaterEq[me: Self](other: Self) -> bool {
    var comp: Ordering =  me.(CompareWith(String).Compare)(other);
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
  let Result:! Type;
  fn Op[me: Self]() -> Result;
}

interface AddWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO: constraint Add { ... }

interface SubWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO: constraint Sub { ... }

interface MulWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO: constraint Mul { ... }

interface ModWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO: constraint Mod { ... }

// Note, these impls use the builtin addition for i32.
external impl i32 as Negate where .Result == i32 {
  fn Op[me: i32]() -> i32 { return -me; }
}
external impl i32 as AddWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 { return me + other; }
}
external impl i32 as SubWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 { return me - other; }
}
external impl i32 as MulWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 { return me * other; }
}
external impl i32 as ModWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 { return me % other; }
}

// ---------------------------------
// Bitwise and bit-shift interfaces.
// ---------------------------------

// Unary `^`.
interface BitComplement {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self]() -> Result;
}

// Binary `&`.
interface BitAndWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO:
// constraint BitAnd {
//   extends BitAndWith(Self) where .Result == Self;
// }

// Binary `|`.
interface BitOrWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO:
// constraint BitOr {
//   extends BitOrWith(Self) where .Result == Self;
// }

// Binary `^`.
interface BitXorWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO:
// constraint BitXor {
//   extends BitXorWith(Self) where .Result == Self;
// }

// Binary `<<`.
interface LeftShiftWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO:
// constraint LeftShift {
//   extends LeftShiftWith(Self) where .Result == Self;
// }

// Binary `>>`.
interface RightShiftWith(U:! Type) {
  // TODO: = Self
  let Result:! Type;
  fn Op[me: Self](other: U) -> Result;
}
// TODO:
// constraint RightShift {
//   extends RightShiftWith(Self) where .Result == Self;
// }

external impl i32 as BitComplement where .Result == i32 {
  fn Op[me: i32]() -> i32 {
    return __intrinsic_int_bit_complement(me);
  }
}
external impl i32 as BitAndWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_and(me, other);
  }
}
external impl i32 as BitOrWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_or(me, other);
  }
}
external impl i32 as BitXorWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 {
    return __intrinsic_int_bit_xor(me, other);
  }
}
external impl i32 as LeftShiftWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 {
    return __intrinsic_int_left_shift(me, other);
  }
}
external impl i32 as RightShiftWith(i32) where .Result == i32 {
  fn Op[me: i32](other: i32) -> i32 {
    return __intrinsic_int_right_shift(me, other);
  }
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

fn Rand(low: i32, high: i32) -> i32{
    return __intrinsic_rand(low,high);
}

class Heap {
  fn New[T:! Type, me: Self](x : T) -> T* {
    return __intrinsic_new(x);
  }
  fn Delete[T:! Type, me: Self](p : T*) {
    __intrinsic_delete(p);
  }
}

var heap: Heap = {};
