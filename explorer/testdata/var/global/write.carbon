// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// Test mutation of a global variable.

var zero: i32 = 1;

fn Main() -> i32 {
  zero = 0;
  return zero;
}

// CHECK:STDOUT: result: 0
