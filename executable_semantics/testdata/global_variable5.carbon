// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test overshadowing of global variable.

var x: i32 = 1;

fn identity(x: i32) {
  return x;
}

fn main() -> i32 {
  return identity(0);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/global_variable5.carbon:15: type error in return
// CHECK: expected: ()
// CHECK: actual: i32
