// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(Int: x, Int: y) -> Int { return x + y; }

fn main() -> Int {
  var (Int, Int): xy = (1, 2);
  // should fail to type-check
  return f(xy);
}
