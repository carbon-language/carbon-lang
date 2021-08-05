// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn fst[T:! Type](x: T, y: T) -> T {
  return x;
}

fn main() -> i32 {
  return fst(0, 1);
}
