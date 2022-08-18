// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-mix-diamond-clash.carbon:34: Member named F1 (declared at {{.*}}/explorer/testdata/mixin/fail-mix-diamond-clash.carbon:16) is being mixed multiple times into C
package ExplorerTest api;

mixin M1 {
  fn F1[me: Self](x: Self) -> Self{
     return x;
  }
}

mixin M2 {
  fn F2() {
  }
  mix M1;
}

mixin M3 {
  mix M1;
  fn F3() {
  }
}

class C {
  mix M2;
  mix M3;
}

fn Main() -> i32 {
  return 0;
}
