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

fn AddInt(a: i32, b: i32) -> i32 {
  var ret: i32 = a + b;
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_missing_declaration.carbon:[[@LINE+1]]: `return var` is not allowed without a returned var defined in scope.
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
