// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/let/fail_function_args.carbon:16: Cannot assign to rvalue 'y'

package ExecutableSemanticsTest api;

fn f((var x: i32, y: i32)) -> i32 {
  x = 0;
  y = 0;
  return x - 1;
}

fn Main() -> i32 {
  var t: auto = (1, 2);
  return f(t);
}
