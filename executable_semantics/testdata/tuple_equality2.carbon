// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var t1: (Int,Int) = (5, 2);
  var t2: (Int,Int) = (5, 4);
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
