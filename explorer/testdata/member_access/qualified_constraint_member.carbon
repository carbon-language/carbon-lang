// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Foo api;
interface A {
  fn F[self: Self]() -> i32;
}
interface B {
  fn G(o: Self) -> i32;
}
alias C = A & B;
class X {
  extend impl as A {
    fn F[self: Self]() -> i32 { return 10 * self.n; }
  }
  extend impl as B {
    fn G(o: Self) -> i32 { return o.n; }
  }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  var w: X = {.n = 2};
  return v.(C.F)() + X.(C.G)(w);
}

// CHECK:STDOUT: result: 12
