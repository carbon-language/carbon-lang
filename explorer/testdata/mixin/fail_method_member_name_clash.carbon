// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

__mixin M {
  fn F[me: Self](x: Self) -> Self{
     return x;
  }
}

class C {
  fn F[me: Self](x: Self) -> Self {
     return x;
  }
  __mix M;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_method_member_name_clash.carbon:[[@LINE+1]]: Member named F (declared at {{.*}}/explorer/testdata/mixin/fail_method_member_name_clash.carbon:[[@LINE-8]]) cannot be mixed into C because it clashes with an existing member with the same name (declared at {{.*}}/explorer/testdata/mixin/fail_method_member_name_clash.carbon:[[@LINE-2]])
}

fn Main() -> i32 {
  return 0;
}
