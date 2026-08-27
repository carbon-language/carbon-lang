// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/sieve.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/sieve.carbon

// --- test.carbon

import Core library "io";

// Compute and return the number of primes less than 1000.

class Sieve {
  impl as Core . UnformedInit { }

  fn Make ( ) -> Sieve {
    returned var s : Sieve ;
    for ( n : i32 in Core . Range ( 1000 ) ) {
      s . is_prime [ n ] = true ;
    }
    return var ;
  }

  fn MarkMultiplesNotPrime ( ref self , p : i32 ) {
    var n : i32 = p * 2 ;
    while ( n < 1000 ) {
      self . is_prime [ n ] = false ;
      n += p ;
    }
  }

  var is_prime : array ( bool , 1000 ) ;
}

fn Run ( ) -> i32 {
  var s : Sieve = Sieve . Make ( ) ;

  var number_of_primes : i32 = 0 ;
  for ( n : i32 in Core . InclusiveRange ( 2 , 999 ) ) {
    if ( s . is_prime [ n ] ) {
      ++ number_of_primes ;
      Core . Print ( n ) ;
      s . MarkMultiplesNotPrime ( n ) ;
    }
  }

  return number_of_primes ;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: import Core library "io";
// CHECK:STDOUT:
// CHECK:STDOUT: // Compute and return the number of primes less than 1000.
// CHECK:STDOUT:
// CHECK:STDOUT: class Sieve {
// CHECK:STDOUT:   impl as Core.UnformedInit {}
// CHECK:STDOUT:
// CHECK:STDOUT:   fn Make() -> Sieve {
// CHECK:STDOUT:     returned var s: Sieve;
// CHECK:STDOUT:     for (n: i32 in Core.Range(1000)) {
// CHECK:STDOUT:       s.is_prime[n] = true;
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return var;
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT:   fn MarkMultiplesNotPrime(ref self, p: i32) {
// CHECK:STDOUT:     var n: i32 = p * 2;
// CHECK:STDOUT:     while (n < 1000) {
// CHECK:STDOUT:       self.is_prime[n] = false;
// CHECK:STDOUT:       n += p;
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT:   var is_prime: array(bool, 1000);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Run() -> i32 {
// CHECK:STDOUT:   var s: Sieve = Sieve.Make();
// CHECK:STDOUT:
// CHECK:STDOUT:   var number_of_primes: i32 = 0;
// CHECK:STDOUT:   for (n: i32 in Core.InclusiveRange(2, 999)) {
// CHECK:STDOUT:     if (s.is_prime[n]) {
// CHECK:STDOUT:       ++number_of_primes;
// CHECK:STDOUT:       Core.Print(n);
// CHECK:STDOUT:       s.MarkMultiplesNotPrime(n);
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT:   return number_of_primes;
// CHECK:STDOUT: }
