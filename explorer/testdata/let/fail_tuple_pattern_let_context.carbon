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

fn Main() -> i32 {
  let (var a: auto, b: auto, c: auto, d: auto) = (1, 2, 3, 4);
  a = 0;
  // should fail
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_tuple_pattern_let_context.carbon:[[@LINE+1]]: Cannot assign to rvalue 'c'
  c = 0;
  return 0;
}
