// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

fn id[T:! Type](x: i32) -> i32 {
  return x;
}

fn main() -> i32 {
  return id(0);
}
