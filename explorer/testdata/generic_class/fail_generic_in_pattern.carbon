// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_generic_in_pattern.carbon:17: Generic binding may not occur in pattern with expected type: T:! i32

package ExplorerTest api;

fn Main() -> i32 {
  var t: auto = 5;
  match (t) {
    case T:! i32 =>
      return 0;
    default =>
      return 1;
  }
}
