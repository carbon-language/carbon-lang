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

fn f(x: i32, y: i32) -> i32 { return x + y; }

fn Main() -> i32 {
  var xy: (i32, i32) = (1, 2);
  // should fail to type-check
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_call_with_tuple.carbon:[[@LINE+1]]: type error in call: '((i32, i32))' is not implicitly convertible to '(i32, i32)'
  return f(xy);
}
