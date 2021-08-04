// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This tests the call-by-value aspect of parameter passing.
// This makes sure that when the value in `x` dies,
// it does not cause the value in `a` to also die.

fn f(x: Int) -> Int {
  return 0;
}

fn main() -> Int {
  var a: Int = 0; var b: Int = 1;
  f(a);
  b = a;
  return b;
}
