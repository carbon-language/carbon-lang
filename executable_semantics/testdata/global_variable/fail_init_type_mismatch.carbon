// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/global_variable/fail_init_type_mismatch.carbon:19: type error in initializer of variable
// CHECK: expected: i32
// CHECK: actual: Bool


package ExecutableSemanticsTest api;

// Test type checking of global variable. Error expected.

var flag: i32 = true;

fn main() -> i32 {
  return 0;
}
