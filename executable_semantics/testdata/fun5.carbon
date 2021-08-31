// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn add(x: i32, y: i32) => x + y;

fn main() -> i32 {
  return add(1, 2) - 3;
}
