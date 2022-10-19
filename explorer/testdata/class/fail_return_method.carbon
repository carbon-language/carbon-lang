// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C {
  fn F() {}
  fn G[me: Self]() {}
}

fn ReturnF() -> auto { return C.F; }
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_return_method.carbon:[[@LINE+1]]: Expected a type, but got member name G
fn ReturnG() -> auto { return C.G; }

fn Main() -> i32 {
  return 0;
}
