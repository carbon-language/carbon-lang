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
  returned var ret: i32 = a + b;
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_missing_return.carbon:[[@LINE+1]]: `return <expression>` is not allowed with a returned var defined in scope: {{.*}}/explorer/testdata/returned_var/fail_missing_return.carbon:14
  return ret;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
