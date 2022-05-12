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

fn Bad[T:! Type](x: {.a: i32, .b: T}) {}

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_implicit_conversion_extra_field.carbon:[[@LINE+1]]: mismatch in field names, source field `c` not in destination type `{.a: i32, .b: T:! Type}`
  Bad({.b = 5, .a = 7, .c = 2});
  return 0;
}
