// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
package ExplorerTest api;

__mixin M1 {
  fn F1[self: Self](x: Self) -> Self{
     return x;
  }
}

__mixin M2 {
  fn F2() {
  }
  __mix M1;
}

__mixin M3 {
  __mix M1;
  fn F3() {
  }
}

class C {
  __mix M2;
  __mix M3;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_mix_diamond_clash.carbon:[[@LINE+1]]: Member named F1 (declared at {{.*}}/explorer/testdata/mixin/fail_mix_diamond_clash.carbon:[[@LINE-18]]) is being mixed multiple times into C
}

fn Main() -> i32 {
  return 0;
}
