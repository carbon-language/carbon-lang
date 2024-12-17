// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/uint";

import library "prelude/types/int_literal";
import library "prelude/operators";

private fn MakeUInt(size: IntLiteral()) -> type = "int.make_type_unsigned";

class UInt(N:! IntLiteral()) {
  adapt MakeUInt(N);
}

impl forall [N:! IntLiteral()] IntLiteral() as ImplicitAs(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: Remove these once ImplicitAs extends As.
impl forall [N:! IntLiteral()] IntLiteral() as As(UInt(N)) {
  fn Convert[self: Self]() -> UInt(N) = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as As(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

impl forall [N:! IntLiteral()] UInt(N) as Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}
impl forall [N:! IntLiteral()] UInt(N) as AddAssign {
  fn Op[addr self: Self*](other: Self) {
    // TODO: Once operator lookup works inside Core.
    // *self = *self + other;
    *self = self->(Add.Op)(other);
  }
}
impl forall [N:! IntLiteral()] UInt(N) as Inc {
  fn Op[addr self: Self*]() {
    // *self += 1;
    self->(AddAssign.Op)(1);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}
impl forall [N:! IntLiteral()] UInt(N) as BitAndAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self & other;
    *self = self->(BitAnd.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as BitComplement {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl forall [N:! IntLiteral()] UInt(N) as BitOr {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}
impl forall [N:! IntLiteral()] UInt(N) as BitOrAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self | other;
    *self = self->(BitOr.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as BitXor {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}
impl forall [N:! IntLiteral()] UInt(N) as BitXorAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self ^ other;
    *self = self->(BitXor.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Div {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}
impl forall [N:! IntLiteral()] UInt(N) as DivAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self / other;
    *self = self->(Div.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Eq {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [N:! IntLiteral()] UInt(N) as LeftShift {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}
impl forall [N:! IntLiteral()] UInt(N) as LeftShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self << other;
    *self = self->(LeftShift.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Mod {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}
impl forall [N:! IntLiteral()] UInt(N) as ModAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self % other;
    *self = self->(Mod.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Mul {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}
impl forall [N:! IntLiteral()] UInt(N) as MulAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self * other;
    *self = self->(Mul.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Negate {
  fn Op[self: Self]() -> Self = "int.snegate";
}

impl forall [N:! IntLiteral()] UInt(N) as Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

impl forall [N:! IntLiteral()] UInt(N) as RightShift {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}
impl forall [N:! IntLiteral()] UInt(N) as RightShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self >> other;
    *self = self->(RightShift.Op)(other);
  }
}

impl forall [N:! IntLiteral()] UInt(N) as Sub {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}
impl forall [N:! IntLiteral()] UInt(N) as SubAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self - other;
    *self = self->(Sub.Op)(other);
  }
}
impl forall [N:! IntLiteral()] UInt(N) as Dec {
  fn Op[addr self: Self*]() {
    // *self -= 1;
    self->(SubAssign.Op)(1);
  }
}
