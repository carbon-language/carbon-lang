// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/uint";

import library "prelude/types/int_literal";
import library "prelude/types/int";
import library "prelude/operators";

private fn MakeUInt(size: IntLiteral()) -> type = "int.make_type_unsigned";

class UInt(N:! IntLiteral()) {
  adapt MakeUInt(N);
}

// Conversions.

impl forall [To:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

impl forall [From:! IntLiteral()] UInt(From) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [To:! IntLiteral()] IntLiteral() as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert_checked";
}

impl forall [From:! IntLiteral()] UInt(From) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Allow as an implicit conversion if To > From.
impl forall [From:! IntLiteral(), To:! IntLiteral()] UInt(From) as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert";
}

// TODO: Allow as an implicit conversion if To > From.
impl forall [From:! IntLiteral(), To:! IntLiteral()] UInt(From) as As(Int(To)) {
  fn Convert[self: Self]() -> Int(To) = "int.convert";
}

// Never implicit.
impl forall [From:! IntLiteral(), To:! IntLiteral()] Int(From) as As(UInt(To)) {
  fn Convert[self: Self]() -> UInt(To) = "int.convert";
}

// Comparisons.

impl forall [N:! IntLiteral()] UInt(N) as Eq {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [N:! IntLiteral()] UInt(N) as Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

// Arithmetic.

impl forall [N:! IntLiteral()] UInt(N) as Add {
  fn Op[self: Self](other: Self) -> Self = "int.uadd";
}

impl forall [N:! IntLiteral()] UInt(N) as Div {
  fn Op[self: Self](other: Self) -> Self = "int.udiv";
}

impl forall [N:! IntLiteral()] UInt(N) as Mod {
  fn Op[self: Self](other: Self) -> Self = "int.umod";
}

impl forall [N:! IntLiteral()] UInt(N) as Mul {
  fn Op[self: Self](other: Self) -> Self = "int.umul";
}

impl forall [N:! IntLiteral()] UInt(N) as Negate {
  fn Op[self: Self]() -> Self = "int.unegate";
}

impl forall [N:! IntLiteral()] UInt(N) as Sub {
  fn Op[self: Self](other: Self) -> Self = "int.usub";
}

// Bitwise operators.

impl forall [N:! IntLiteral()] UInt(N) as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

impl forall [N:! IntLiteral()] UInt(N) as BitComplement {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl forall [N:! IntLiteral()] UInt(N) as BitOr {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

impl forall [N:! IntLiteral()] UInt(N) as BitXor {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

impl forall [N:! IntLiteral()] UInt(N) as LeftShift {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

impl forall [N:! IntLiteral()] UInt(N) as RightShift {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// Compound assignments.

// TODO: Once we can lower specific functions, make these generic.
// Each function has a commented out generic signature.

// impl forall [N:! IntLiteral()] UInt(N) as AddAssign {
impl u32 as AddAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self + other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as BitAndAssign {
impl u32 as BitAndAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self & other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as BitOrAssign {
impl u32 as BitOrAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self | other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as BitXorAssign {
impl u32 as BitXorAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self ^ other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as DivAssign {
impl u32 as DivAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self / other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as LeftShiftAssign {
impl u32 as LeftShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self << other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as ModAssign {
impl u32 as ModAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self % other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as MulAssign {
impl u32 as MulAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self * other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as RightShiftAssign {
impl u32 as RightShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self >> other;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as SubAssign {
impl u32 as SubAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self - other;
  }
}

// Increment and decrement.

// impl forall [N:! IntLiteral()] UInt(N) as Dec {
impl u32 as Dec {
  fn Op[addr self: Self*]() {
    *self -= 1;
  }
}

// impl forall [N:! IntLiteral()] UInt(N) as Inc {
impl u32 as Inc {
  fn Op[addr self: Self*]() {
    *self += 1;
  }
}
