// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:23: Member named F (declared at {{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:17) cannot be mixed into C because it clashes with an existing member with the same name (declared at {{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:21)

package ExplorerTest api;

__mixin M {
  fn F[me: Self](x: Self) -> Self{
     return x;
  }
}

class C {
  let F: i32 = 0;
  __mix M;
}

fn Main() -> i32 {
  return 0;
}
