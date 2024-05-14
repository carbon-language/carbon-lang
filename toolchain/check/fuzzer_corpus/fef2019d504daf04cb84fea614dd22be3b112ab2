// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 3

package Foo api;

class X {
  fn F[self: Self](o: Self) -> Self { return {.n = self.n + o.n}; }
  var n: i32;
}

fn Main() -> i32 {
  return {.n = 1}.(X.F)({.n = 2}).n;
}
