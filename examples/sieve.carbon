// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

// Compute and return the number of primes less than 1000.

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
    var n: i32 = p * 2;
    while (n < 1000) {
      self->is_prime[n] = false;
      n += p;
    }
  }

  var is_prime: array(bool, 1000);
}

fn Run() -> i32 {
  var s: Sieve = Sieve.Make();

  var number_of_primes: i32 = 0;
  var n: i32 = 2;
  while (n < 1000) {
    if (s.is_prime[n]) {
      ++number_of_primes;
      Core.Print(n);
      s.MarkMultiplesNotPrime(n);
    }
    ++n;
  }

  return number_of_primes;
}
