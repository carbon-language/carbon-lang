// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s
// RUN: ../executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s

package ExecutableSemanticsTest api;

// Test mutation of a global variable.

var zero: i32 = 1;

fn main() -> i32 {
  zero = 0;
  return zero;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
