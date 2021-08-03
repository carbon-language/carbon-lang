// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var x: Int = 2;
  while (true) {
    if (x == 0) {
      break;
    } else {
      x = x - 1;
    }
  }
  return x;
}
