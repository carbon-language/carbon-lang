// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This tests the call-by-value aspect of parameter passing.
// This makes sure that when the value in `x` dies,
// it does not cause the value in `a` to also die.

fn f(Int: x) -> Int {
  return 0;
}

fn main() -> Int {
  var Int: a = 0; var Int: b = 1;
  f(a);
  b = a;
  return b;
}
