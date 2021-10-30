// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/class/fail_field_missing.carbon:20: type error in name binding: '{.x: i32}' is not implicitly convertible to 'class Point'

package ExecutableSemanticsTest api;

class Point {
  var x: i32;
  var y: i32;
}

fn main() -> i32 {
  var p: Point = {.x = 1};
  return p.x - 1;
}
