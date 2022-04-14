// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: RUNTIME ERROR: {{.*}}/executable_semantics/testdata/array/fail_index.carbon:16: index 2 out of range in (0, 1)

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var x: [i32; 2] = (0, 1);
  return x[2];
}
