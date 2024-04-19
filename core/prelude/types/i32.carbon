// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/i32" api;

import library "prelude/operators";

// TODO: Remove these once prelude/operators re-exports their contents.
import library "prelude/operators/arithmetic";
import library "prelude/operators/bitwise";
import library "prelude/operators/comparison";

impl i32 as Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}
impl i32 as AddAssign {
  fn Op[addr self: Self*](other: Self) {
    // TODO: Once operator lookup works inside Core.
    // *self = *self + other;
    *self = self->(Add.Op)(other);
  }
}
impl i32 as Inc {
  fn Op[addr self: Self*]() {
    // *self += 1;
    self->(AddAssign.Op)(1);
  }
}

impl i32 as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}
impl i32 as BitAndAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self & other;
    *self = self->(BitAnd.Op)(other);
  }
}

impl i32 as BitComplement {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl i32 as BitOr {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}
impl i32 as BitOrAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self | other;
    *self = self->(BitOr.Op)(other);
  }
}

impl i32 as BitXor {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}
impl i32 as BitXorAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self ^ other;
    *self = self->(BitXor.Op)(other);
  }
}

impl i32 as Div {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}
impl i32 as DivAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self / other;
    *self = self->(Div.Op)(other);
  }
}

impl i32 as Eq {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl i32 as LeftShift {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}
impl i32 as LeftShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self << other;
    *self = self->(LeftShift.Op)(other);
  }
}

impl i32 as Mod {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}
impl i32 as ModAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self % other;
    *self = self->(Mod.Op)(other);
  }
}

impl i32 as Mul {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}
impl i32 as MulAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self * other;
    *self = self->(Mul.Op)(other);
  }
}

impl i32 as Negate {
  fn Op[self: Self]() -> Self = "int.snegate";
}

impl i32 as Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

impl i32 as RightShift {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}
impl i32 as RightShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self >> other;
    *self = self->(RightShift.Op)(other);
  }
}

impl i32 as Sub {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}
impl i32 as SubAssign {
  fn Op[addr self: Self*](other: Self) {
    // *self = *self - other;
    *self = self->(Sub.Op)(other);
  }
}
impl i32 as Dec {
  fn Op[addr self: Self*]() {
    // *self -= 1;
    self->(SubAssign.Op)(1);
  }
}
