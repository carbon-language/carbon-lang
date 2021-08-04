// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn Id(t: Type) => t;

// Test non-trivial type expression in variable declaration statement.

fn main() -> Int {
  var x: Id(Int) = 0;
  return x;
}
