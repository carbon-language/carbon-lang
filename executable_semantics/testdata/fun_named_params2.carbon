// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

fn f(x: i32, .d = y: i32, z: i32, .e = a: i32) => (x + y) - (z + a);

fn main() -> i32 {
  return 0;
}
