// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn swap[T:! Type, U:! Type](tuple: (T, U)) -> (U, T) {
  return (tuple[1], tuple[0]);
}

fn main() -> Int {
  return swap((0, true))[1];
}
