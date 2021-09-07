// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  var t1: (i32,i32) = (5, 2);
  var t2: (i32,i32) = (5, 4);
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
