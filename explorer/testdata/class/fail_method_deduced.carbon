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

fn H[T:! Type](x: T) {}

fn Main() -> i32 {
  H(C.F);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_method_deduced.carbon:[[@LINE+1]]: member name G can only be used in a member access or alias
  H(C.G);
  return 0;
}
