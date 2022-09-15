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
  __mix M1;
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_recursive_mixing.carbon:[[@LINE+1]]: Member named F1 (declared at {{.*}}/explorer/testdata/mixin/fail_recursive_mixing.carbon:15) is being mixed multiple times into M1
}

class C {
  __mix M1;
}

fn Main() -> i32 {
  return 0;
}
