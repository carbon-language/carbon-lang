// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

choice i32s {
  None,
  One(i32),
  Two(i32,i32)
}

fn main() -> i32 {
  var x: auto = i32s.None();
  var y: auto = i32s.One(42);
  var n: auto = 0;
  match (y) {
    case i32s.None =>
      n = n + 2;
    case i32s.One(x: auto) =>
      n = x + 1 - 42;
    case i32s.Two(a: auto, b: auto) =>
      n = 2;
  }
  match (x) {
    case i32s.One(x: auto) =>
      n = x + 2;
    case i32s.None() =>
      n = n - 1;
    case i32s.Two(x: auto, y: auto) =>
      n = 5;
  }
  return n;
}

// Test some alternate syntaxes
choice Morei32s {
  None(),
  One(i32),
  Two(i32,i32),
}
