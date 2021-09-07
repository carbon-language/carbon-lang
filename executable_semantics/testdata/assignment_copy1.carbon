// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test that assignment performs a copy and does not create an alias.

fn main() -> i32 {
  var x: i32 = -1;
  {
     var y: i32 = 0;
     x = y;
     // y dies here
  }
  return x;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
