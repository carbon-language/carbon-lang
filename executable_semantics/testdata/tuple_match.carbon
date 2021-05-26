// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var auto: t = (5, 2);
  match (t) {
    case (auto: a, auto: b) =>
      return a + b - 7;
  }
}
