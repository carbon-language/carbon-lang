// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/let/fail_tuple_pattern_let_context_nested.carbon:15: syntax error, unexpected LET

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  let (var a: auto, b: auto, c:auto, var (d: auto, let e:auto)) = (1, 2, 3, (4, 5));
  return 0;
}
