// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var t: auto = (1, 2, 3, 4);
  match (t) {
    case (_: Int, _: auto, x: Int, y: auto) =>
      return y - x - 1;
  }
}
