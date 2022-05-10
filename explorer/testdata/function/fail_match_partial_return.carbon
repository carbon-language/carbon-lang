// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_match_partial_return.carbon:17: control-flow reaches end of function that provides a `->` return type without reaching a return statement

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  match (0) {
    case 1 => { x = 1; }
    case _: auto => { return 0; }
  }
}
