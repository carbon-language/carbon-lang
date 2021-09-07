// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest library "Foo" api;

fn main() -> i32 {
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
