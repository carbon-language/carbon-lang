// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compute and return the number of primes less than 1000.

import Core library "operators";

// TODO: Copied from i32.carbon.
// Remove the following, once we do cross-file impl lookup.
// import Core library "i32";

impl i32 as Core.Add {
  fn Op[self: Self](other: Self) -> Self = "int.add";
}
impl i32 as Core.AddAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self + other;
  }
}
impl i32 as Core.Inc {
  fn Op[addr self: Self*]() {
    *self += 1;
  }
}

impl i32 as Core.BitAnd {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}
impl i32 as Core.BitAndAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self & other;
  }
}

impl i32 as Core.BitComplement {
  fn Op[self: Self]() -> Self = "int.complement";
}

impl i32 as Core.BitOr {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}
impl i32 as Core.BitOrAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self | other;
  }
}

impl i32 as Core.BitXor {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}
impl i32 as Core.BitXorAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self ^ other;
  }
}

impl i32 as Core.Div {
  fn Op[self: Self](other: Self) -> Self = "int.div";
}
impl i32 as Core.DivAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self / other;
  }
}

impl i32 as Core.Eq {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl i32 as Core.LeftShift {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}
impl i32 as Core.LeftShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self << other;
  }
}

impl i32 as Core.Mod {
  fn Op[self: Self](other: Self) -> Self = "int.mod";
}
impl i32 as Core.ModAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self % other;
  }
}

impl i32 as Core.Mul {
  fn Op[self: Self](other: Self) -> Self = "int.mul";
}
impl i32 as Core.MulAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self * other;
  }
}

impl i32 as Core.Negate {
  fn Op[self: Self]() -> Self = "int.negate";
}

impl i32 as Core.Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

impl i32 as Core.RightShift {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}
impl i32 as Core.RightShiftAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self >> other;
  }
}

impl i32 as Core.Sub {
  fn Op[self: Self](other: Self) -> Self = "int.sub";
}
impl i32 as Core.SubAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self - other;
  }
}
impl i32 as Core.Dec {
  fn Op[addr self: Self*]() {
    *self -= 1;
  }
}

// ---

fn Run() -> i32 {
  var is_prime: [bool; 1000];

  // TODO: `for` loop.
  var n: i32 = 0;
  while (n < 1000) {
    is_prime[n] = true;
    ++n;
  }

  var number_of_primes: i32 = 0;
  n = 2;
  while (n < 1000) {
    if (is_prime[n]) {
      ++number_of_primes;
      var k: i32 = 2 * n;
      while (k < 1000) {
        is_prime[k] = false;
        k += n;
      }
    }
    ++n;
  }

  return number_of_primes;
}
