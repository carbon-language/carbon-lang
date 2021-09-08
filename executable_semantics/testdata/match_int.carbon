// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s
// RUN: ../executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  var t: auto = 5;
  match (t) {
    case 5 =>
      return 0;
    default =>
      return 1;
  }
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
