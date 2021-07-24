// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn map[T:! Type](f: fnty (T) -> T, tuple: (T, T)) -> (T, T) {
  return (f(tuple[0]), f(tuple[1]));
}

fn inc(x: Int) -> Int { return x + 1; }

fn main() -> Int {
  return map(inc, (0, 2))[0];
}
