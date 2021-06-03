// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn add(x: Int, y: Int) => x + y;

fn main() -> Int {
  return add(1, 2) - 3;
}
