// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package Foo api;
interface A { fn F[self: Self]() -> i32; }
class X {
  impl as A {
    fn F[self: Self]() -> i32 { return 1; }
  }
}
fn Main() -> i32 {
  var a: X = {};
  return a.(X.(A.F))();
}
