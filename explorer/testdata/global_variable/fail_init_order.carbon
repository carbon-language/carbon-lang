// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/global_variable/fail_init_order.carbon:17: could not find `y: i32`

package ExplorerTest api;

// Test that a global variable may not depend on a later global.
// Error expected.

var x: i32 = y;

var y: i32 = 0;

fn Main() -> i32 {
  return x;
}
