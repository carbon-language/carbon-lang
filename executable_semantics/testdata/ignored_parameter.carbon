// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn ReturnSecond(_: Int, x: Int) -> Int {
  return x;
}

fn main() -> Int {
  return ReturnSecond(1, 0);
}
