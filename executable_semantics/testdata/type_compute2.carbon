// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn Id(t: Type) => t;

// Test non-trivial type expression in return type.

fn main() -> Id(i32) {
  return 0;
}
