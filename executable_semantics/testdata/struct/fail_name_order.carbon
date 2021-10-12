// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/struct/fail_name_order.carbon:17: Type pattern '{.x: i32, .y: i32}' does not match actual type '{.y: i32, .x: i32}'

package ExecutableSemanticsTest api;

// Test the that field order matters for structs.

fn main() -> i32 {
  var t: {.x: i32, .y: i32} = {.y = 2, .x = 3};
  return 0;
}
