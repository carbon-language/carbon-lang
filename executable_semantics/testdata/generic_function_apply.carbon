// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn apply[T:! Type, U:! Type](f: fnty (T) -> U, x: T) -> U {
  return f(x);
}

fn positive(x: Bool) -> Int {
  if (x) {
    return 2;
  } else {
    return -2;
  }
}

fn main() -> Int {
  return apply(positive, false);
}
