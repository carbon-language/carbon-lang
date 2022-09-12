// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
package ExplorerTest api;

__mixin M1 {
  fn F1[me: Self](x: Self) -> Self{
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
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:[[@LINE+1]]: Member named F1 (declared at {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:20) cannot be mixed into C because it clashes with an existing member with the same name (declared at {{.*}}/explorer/testdata/mixin/fail_mix_members_clash.carbon:15)
}

fn Main() -> i32 {
  return 0;
}
