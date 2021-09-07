// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn F() -> auto {
  return 0;
}

fn main() -> i32 {
  return F();
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/return_auto.carbon:10: syntax error, unexpected AUTO
