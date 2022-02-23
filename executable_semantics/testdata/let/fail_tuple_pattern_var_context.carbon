// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/let/fail_tuple_pattern_var_context.carbon:18: Cannot assign to rvalue 'c'

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var (var a: auto, b: auto, let c:auto, d: auto) = (1, 2, 3, 4);
  a = 0;
  b = 0;
  c = 0;
  return 0;
}
