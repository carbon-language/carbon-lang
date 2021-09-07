// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test multiple arguments
fn f(x: i32, y: i32) -> i32 {
  return x + y;
}

fn main() -> i32 {
  return f(2,3) - 5;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
