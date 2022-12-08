// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 1
// CHECK:STDOUT: 1
// CHECK:STDOUT: 1
// CHECK:STDOUT: 1
// CHECK:STDOUT: result: 0

package Foo api;
class X {
  fn F[self: Self]() -> i32 { return self.n; }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  let p: X* = &v;
  Print("{0}", p->n);
  Print("{0}", p->(X.n));
  Print("{0}", p->F());
  Print("{0}", p->(X.F)());
  return 0;
}
