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
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-recursive-mixing.carbon:[[@LINE+1]]: 'M1' is not usable until after it has been completely declared
  __mix M1;
}

class C {
  __mix M1;
}

fn Main() -> i32 {
  return 0;
}
