// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A {}
impl A as ImplicitAs(i32) {
  fn Convert[me: Self]() -> i32 { return 1; }
}

fn Main() -> i32 {
  var a: A = {};

  match ((a, a)) {
    case (0, n: i32) => { return 1; }
    case (n: i32, 0) => { return 1; }
    case (1, 1) => { return 0; }
  }

  return 1;
}
