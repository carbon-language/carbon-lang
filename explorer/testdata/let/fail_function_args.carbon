// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_function_args.carbon:16: Cannot assign to rvalue 'y'

package ExplorerTest api;

fn f((var x: i32, y: i32)) -> i32 {
  x = 0;
  y = 0;
  return x - 1;
}

fn Main() -> i32 {
  var t: auto = (1, 2);
  return f(t);
}
