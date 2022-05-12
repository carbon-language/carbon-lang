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
  var x: i32 = 0;
  match(0) {
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_match_no_return.carbon:[[@LINE+1]]: control-flow reaches end of function that provides a `->` return type without reaching a return statement
    case 1 => { x = 1; }
    case _: auto => { x = 2; }
  }
}
