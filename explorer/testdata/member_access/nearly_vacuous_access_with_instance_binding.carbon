// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 1

package Foo api;
interface A { fn F[me: Self]() -> i32; }
class X {
  impl as A {
    fn F[me: Self]() -> i32 { return 1; }
  }
}
fn Main() -> i32 {
  var a: X = {};
  return a.(X.(A.F))();
}
