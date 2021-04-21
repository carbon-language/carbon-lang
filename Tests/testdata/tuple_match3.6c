// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test matching of a tuple inside a tuple.

fn main() -> Int {
  var auto: t = ((1,2),(3,4));
  match (t) {
    case ((auto: a, auto: b), auto: c) =>
      return a - b + c[0] - c[1] + 2;
  }
}
