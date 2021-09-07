// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test a global variable depending on another global.

var x: i32 = 0;

var y: i32 = x;

fn main() -> i32 {
  return y;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
