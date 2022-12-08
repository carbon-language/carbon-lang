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
  fn F1() {
  }
}

class C {
  __mix M1;
  __mix M2;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:[[@LINE+1]]: Member named F1 (declared at {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:[[@LINE-6]]) cannot be mixed into C because it clashes with an existing member with the same name (declared at {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:[[@LINE-11]])
}

fn Main() -> i32 {
  return 0;
}
