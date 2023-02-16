// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package Foo api;

interface HasF {
  fn F[self: Self](o: Self) -> Self;
}

class X {
  impl as HasF {
    fn F[self: Self](o: Self) -> Self { return {.n = self.n + o.n}; }
  }
  var n: i32;
}

class Y {
  var m: i32;
  impl as ImplicitAs(X) {
    fn Convert[self: Self]() -> X { return {.n = self.m}; }
  }
}

fn Main() -> i32 {
  var x: X = {.n = 1};
  var y: Y = {.m = 2};
  return y.(X.(HasF.F))(x).n;
}
