// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> Int {
  var Int: x = 0;
  {
    var Int: x = 1;
  }
  return x;
}
