// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
package ExplorerTest api;

__mixin M1 {
  fn F1[me: Self](x: Self) -> Self{
     return x;
  }
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_recursive_mixing.carbon:[[@LINE+1]]: incomplete mixin `M1` used in mix declaration
  __mix M1;
}

class C {
  __mix M1;
}

fn Main() -> i32 {
  return 0;
}
