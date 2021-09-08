// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not ../executable_semantics %s 2>&1 | FileCheck %s

package ExecutableSemanticsTest api;

fn id[T:! Type](x: i32) -> i32 {
  return x;
}

fn main() -> i32 {
  return id(0);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/generic_function_fail2.carbon:14: could not deduce type argument for type parameter T
