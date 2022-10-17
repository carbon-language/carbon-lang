// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 5

package ExplorerTest api;

class A {
  var n: i32;
}
impl i32 as ImplicitAs(A) {
  fn Convert[me: Self]() -> A { return {.n = me}; }
}

var a: A = 5;

fn Main() -> i32 {
  return a.n;
}
