// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1133

package ExplorerTest api;

interface I(T:! Type) {
  fn F[me: Self](t: T, o: Self) -> Self;
}

class X(U:! Type) {
  impl as I(U) {
    fn F[me: Self](u: U, o: Self) -> Self { return {.m = u, .n = me.n + o.n}; }
  }
  var m: U;
  var n: i32;
}

fn Run[V:! Type](x: V) -> (V, V, i32, i32) {
  var v: X(V) = {.m = x, .n = 1};
  var w: X(V) = {.m = x, .n = 2};
  // OK, know that `X(V)` implements `I(V)` from the `impl` in the class.
  var call: X(V) = v.(X(V).(I(V).F))(x, w);
  return (call.(X(V).m), call.m, call.(X(V).n), call.n);
}

fn Main() -> i32 {
  var (a: i32, b: i32, c: i32, d: i32) = Run(1);
  return a * 1000 + b * 100 + c * 10 + d;
}
