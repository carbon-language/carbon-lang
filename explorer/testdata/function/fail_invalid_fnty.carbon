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

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_invalid_fnty.carbon:[[@LINE+3]]: type error in negation
// CHECK: expected: i32
// CHECK: actual: Bool
fn f(g: __Fn(-true) -> true) {
}

fn Main() -> i32 {
  return 0;
}
