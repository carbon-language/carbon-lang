// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(x: Int, .d = y: Int) => x + y;

fn main() -> Int {
  return f(1, .d = 2) - 3;
}
