// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 5

package ExplorerTest api;

class A { var n: i32; }

external impl A as ImplicitAs(i32) {
  fn Convert[me: Self]() -> i32 { return me.n; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  return a as i32;
}
