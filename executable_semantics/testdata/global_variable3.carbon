// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test type checking of global variable. Error expected.

var flag: i32 = true;

fn main() -> i32 {
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/global_variable3.carbon:12: type error in initializer of variable
// CHECK: expected: i32
// CHECK: actual: Bool
