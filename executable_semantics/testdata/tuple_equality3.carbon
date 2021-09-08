// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not ../executable_semantics %s 2>&1 2>&1 | FileCheck %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  var t1: (i32,i32) = (5, 2);
  var t2: (i32,) = (5,);
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/tuple_equality3.carbon:12: type error in ==
// CHECK: expected: (0 = i32, 1 = i32)
// CHECK: actual: (0 = i32)
