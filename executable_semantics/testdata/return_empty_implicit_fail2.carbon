// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not ../executable_semantics %s 2>&1 | FileCheck %s

package ExecutableSemanticsTest api;

fn F() -> () {
  return;
}

fn main() -> i32 {
  F();
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/return_empty_implicit_fail2.carbon:10: return; should provide a return value, to match the function's signature.
