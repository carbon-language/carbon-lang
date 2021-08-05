// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn main() -> i32 {
  var t: (i32, .x = i32) = (3, .x = 2);
  return t.x + 1 - t[0];
}
