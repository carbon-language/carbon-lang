// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn Id(Type: t) => t;

// Test non-trivial type expression in parameter type.

fn f(Id(Int): x) -> Int {
  return x;
}

fn main() -> Int {
  return f(0);
}
