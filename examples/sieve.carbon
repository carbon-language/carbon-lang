// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compute and return the number of primes less than 1000.

import Core library "prelude/operators/arithmetic";
import Core library "prelude/operators/comparison";

// TODO: Copied from core/prelude/types/i32.carbon.
// Because we don't deduplicate interfaces, the implementations in that file are
// treated as implementing a different interface from the one we import above.
// Remove the following, once we deduplicate interfaces.

impl i32 as Core.Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
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

impl i32 as Core.Div {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
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

impl i32 as Core.Mod {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}
impl i32 as Core.ModAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self % other;
  }
}

impl i32 as Core.Mul {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}
impl i32 as Core.MulAssign {
  fn Op[addr self: Self*](other: Self) {
    *self = *self * other;
  }
}

impl i32 as Core.Negate {
  fn Op[self: Self]() -> Self = "int.snegate";
}

impl i32 as Core.Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

impl i32 as Core.Sub {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
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

class Sieve {
  fn Make() -> Sieve {
    returned var s: Sieve;

    // TODO: `for` loop.
    var n: i32 = 0;
    while (n < 1000) {
      s.is_prime[n] = true;
      ++n;
    }

    return var;
  }

  fn MarkMultiplesNotPrime[addr self: Self*](p: i32) {
    var n: i32 = 2 * p;
    while (n < 1000) {
      self->is_prime[n] = false;
      n += p;
    }
  }

  var is_prime: [bool; 1000];
}

fn Run() -> i32 {
  var s: Sieve = Sieve.Make();

  var number_of_primes: i32 = 0;
  var n: i32 = 2;
  while (n < 1000) {
    if (s.is_prime[n]) {
      ++number_of_primes;
      s.MarkMultiplesNotPrime(n);
    }
    ++n;
  }

  return number_of_primes;
}
