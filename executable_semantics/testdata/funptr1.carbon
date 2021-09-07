// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn add1(x: i32) -> i32 {
  return x + 1;
}

fn main() -> i32 {
  var f: fnty(i32)->i32 = add1;
  return f(-1);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
