// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test matching of a tuple inside a tuple.

fn main() -> Int {
  var t: auto = ((1,2),(3,4));
  match (t) {
    case ((a: auto, b: auto), c: auto) =>
      return a - b + c[0] - c[1] + 2;
  }
}
