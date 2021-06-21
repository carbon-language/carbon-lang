// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var t1: (Int,) = (5,);
  var t2: (Int, Int) = (2, 3,);
  return t1[0] - t2[0] - t2[1];
}
