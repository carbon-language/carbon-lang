// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

// Test matching of a tuple inside a tuple.

fn main() -> i32 {
  var t: auto = ((1,2),(3,4));
  match (t) {
    case ((a: auto, b: auto), c: auto) =>
      return a - b + c[0] - c[1] + 2;
  }
  return 1;
}
