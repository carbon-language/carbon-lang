// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

__mixin M {
  fn F[me: Self](x: Self) -> Self{
     return x;
  }
}

class C {
  let F: i32 = 0;
  __mix M;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_field_member_name_clash.carbon:[[@LINE+1]]: Member named F (declared at {{.*}}/explorer/testdata/mixin/fail_field_member_name_clash.carbon:[[@LINE-6]]) cannot be mixed into C because it clashes with an existing member with the same name (declared at {{.*}}/explorer/testdata/mixin/fail_field_member_name_clash.carbon:[[@LINE-2]])
}

fn Main() -> i32 {
  return 0;
}
