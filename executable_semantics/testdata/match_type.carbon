// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var t: auto = fnty (Int,Int);
  var x: Int = 0;
  match (t) {
    case z: Type =>
      x = x + 1;
  }
  match (t) {
    case fnty(a: Type,b: Type) =>
      x = x - 1;
  }
  return x;
}
