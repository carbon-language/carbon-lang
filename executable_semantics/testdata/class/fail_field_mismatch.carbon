// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/class/fail_field_mismatch.carbon:23: type error in call
// CHECK: expected: (x = i32, y = i32)
// CHECK: actual: (x = i32, z = i32)


package ExecutableSemanticsTest api;

class Point {
  var x: i32;
  var y: i32;
}

fn main() -> i32 {
  return Point(.x = 1, .z = 2).x - 1;
}
