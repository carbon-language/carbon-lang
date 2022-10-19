// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package Foo api;

class X {
  fn F[me: Self](o: Self) -> Self { return {.n = me.n + o.n}; }
  var n: i32;
}

class Y {
  var m: i32;
  impl as ImplicitAs(X) {
    fn Convert[me: Self]() -> X { return {.n = me.m}; }
  }
}

fn Main() -> i32 {
  var y1: Y = {.m = 1};
  var y2: Y = {.m = 2};
  return y1.(X.F)(y2).n;
}
