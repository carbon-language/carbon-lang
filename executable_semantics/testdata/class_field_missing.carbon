// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

class Point {
  var x: i32;
  var y: i32;
}

fn main() -> i32 {
  return Point(.x = 1).x - 1;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/class_field_missing.carbon:16: type error in call
// CHECK: expected: (x = i32, y = i32)
// CHECK: actual: (x = i32)
