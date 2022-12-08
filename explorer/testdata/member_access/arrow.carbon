// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 4

package Foo api;
class X {
  fn F[self: Self]() -> i32 { return self.n; }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  let p: X* = &v;
  return p->n + p->(X.n) + p->F() + p->(X.F)();
}
