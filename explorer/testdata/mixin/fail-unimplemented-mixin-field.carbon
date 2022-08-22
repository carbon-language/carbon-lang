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
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-unimplemented-mixin-field.carbon:[[@LINE+1]]: syntax error, unexpected VAR, expecting FN or MIX or RIGHT_CURLY_BRACE
  var v: i32 = 42;
}

class C {
 __mix M1;
}

fn Main() -> i32 {
  var c: C = {.v = 0};
  return c.v;
}
