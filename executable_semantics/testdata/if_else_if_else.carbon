// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  if (0 == 1) {
    return 1;
  } else if (0 == 2) {
    return 2;
  } else {
    return 0;
  }

  return 1;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
