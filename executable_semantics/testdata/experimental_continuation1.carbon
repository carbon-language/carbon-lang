// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s
// RUN: ../executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s

package ExecutableSemanticsTest api;

// Test that creating a continuation doesn't do anything.

fn main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
  }
  return x;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
