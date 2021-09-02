// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

fn Id(t: Type) => t;

// Test non-trivial type expression in parameter type.

fn f(x: Id(i32)) -> i32 {
  return x;
}

fn main() -> i32 {
  return f(0);
}
