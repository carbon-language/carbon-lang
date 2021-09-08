// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not ../executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s
// RUN: not ../executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s

package ExecutableSemanticsTest api;

fn F() -> () {
}

fn main() -> i32 {
  F();
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/return_empty_implicit_fail1.carbon:13: control-flow reaches end of function that provides a `->` return type without reaching a return statement
