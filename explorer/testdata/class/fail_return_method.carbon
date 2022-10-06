// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class C {
  fn F() {}
  fn G[me: Self]() {}
}

fn ReturnF() -> auto { return C.F; }
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_return_method.carbon:[[@LINE+1]]: Expected a type, but got member name G
fn ReturnG() -> auto { return C.G; }

fn Main() -> i32 {
  return 0;
}
