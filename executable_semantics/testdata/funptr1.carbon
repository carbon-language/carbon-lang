// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn add1(x: Int) -> Int {
  return x + 1;
}

fn main() -> Int {
  var f: fnty(Int)->Int = add1;
  return f(-1);
}
