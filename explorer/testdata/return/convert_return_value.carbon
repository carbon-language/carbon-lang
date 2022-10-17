// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 42

package ExplorerTest api;

class A {
  impl as ImplicitAs(i32) {
    fn Convert[me: Self]() -> i32 { return 42; }
  }
}

fn Main() -> i32 {
  var a: A = {};
  return a;
}
