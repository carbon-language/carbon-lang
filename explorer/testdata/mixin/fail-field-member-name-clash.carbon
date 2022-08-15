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

mixin M {
  fn F[me: Self](x: Self) -> Self{
     return x;
  }
}

class C {
  let F: i32 = 0;
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:[[@LINE+1]]: Ambiguous member name 'F' ({{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:20) while mixing with the mixin M. (Clashes with mixin member declared at {{.*}}/explorer/testdata/mixin/fail-field-member-name-clash.carbon:16)
  mix M;
}

fn Main() -> i32 {
  return 0;
}
