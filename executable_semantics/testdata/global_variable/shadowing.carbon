// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/global_variable/shadowing.carbon:22: type error in return
// CHECK: expected: ()
// CHECK: actual: i32

package ExecutableSemanticsTest api;

// Test overshadowing of global variable.

var x: i32 = 1;

fn identity(x: i32) {
  return x;
}

fn main() -> i32 {
  return identity(0);
}
