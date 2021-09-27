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

// Test matching of a struct inside a struct.

fn main() -> i32 {
  var t: auto = {.x = {.a = 1, .b = 2}, .y = {.c = 3, .d = 4}};
  match (t) {
    case {.x = {.a = p: auto, .b = q: auto}, .y = r: auto} =>
      return p - q + r.c - r.d + 2;
  }
}
