// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface X {
  fn F() -> Type;
}

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_use_symbolic_member.carbon:[[@LINE+1]]: member lookup for F in symbolic witness for T:! X
fn G[T:! X]() -> T.F() {
  return {};
}

fn Main() -> i32 {
  return 0;
}
