// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

// Test matching with a mixture of positional and named fields.

fn main() -> i32 {
  var t: auto = (2, .x = 5);
  match (t) {
    case (a: auto, .x = b: auto) =>
      return a - b + 3;
  }
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
