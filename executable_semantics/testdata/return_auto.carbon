// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s


package ExecutableSemanticsTest api;

fn F() -> auto {
  return 0;
}

fn main() -> i32 {
  return F();
}

// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/return_auto.carbon:13: syntax error, unexpected AUTO
