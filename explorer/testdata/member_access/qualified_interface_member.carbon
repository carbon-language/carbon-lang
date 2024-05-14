// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Foo api;
interface A {
  fn F[self: Self](o: Self) -> Self;
}
class X {
  extend impl as A {
    fn F[self: Self](o: Self) -> Self { return {.n = self.n + o.n}; }
  }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  var w: X = {.n = 2};
  return v.(A.F)(w).n;
}

// CHECK:STDOUT: result: 3
