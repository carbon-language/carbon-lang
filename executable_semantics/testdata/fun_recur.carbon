// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(x: Int) -> Int {
  if (x == 0) {
    return x;
  } else {
    return f(x - 1);
  }
}

fn main() -> Int {
  return f(2);
}
