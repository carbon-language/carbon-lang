// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12

package ExplorerTest api;

class A { var n: i32; }

external impl A as Add {
  fn Op[self: Self](rhs: A) -> A { return {.n = self.n + rhs.n}; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  var b: A = {.n = 7};
  return (a + b).n;
}
