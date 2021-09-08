// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s


package ExecutableSemanticsTest api;

fn main() -> i32 {
  // error
  x : i32;
  return 1;
}

// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/pattern_variable_fail.carbon:15: syntax error, unexpected COLON
