// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn id[T:! Type](x: T) -> T {
  return x + 0;
}

fn main() -> i32 {
  return id(0);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/generic_function_fail3.carbon:11: type error in addition(1)
// CHECK: expected: i32
// CHECK: actual: T
