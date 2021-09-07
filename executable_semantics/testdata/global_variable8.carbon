// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test that a global variable may not depend on a later global.
// Error expected.

var x: i32 = y;

var y: i32 = 0;

fn main() -> i32 {
  return x;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: RUNTIME ERROR: executable_semantics/testdata/global_variable8.carbon:13: could not find `y`
