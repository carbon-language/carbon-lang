// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not ../executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s
// RUN: not ../executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s

package ExecutableSemanticsTest api;

fn f(x: i32, y: i32) -> i32 { return x + y; }

fn main() -> i32 {
  var xy: (i32, i32) = (1, 2);
  // should fail to type-check
  return f(xy);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/fun6_fail_type.carbon:17: type error in call
// CHECK: expected: (0 = i32, 1 = i32)
// CHECK: actual: (0 = (0 = i32, 1 = i32))
