// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> i32 {
  var x: auto = 2;
  while (not (x == 0)) {
    x = x - 1;
  }
  return x;
}
