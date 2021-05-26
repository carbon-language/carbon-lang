// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(x: Int, .d = y: Int, z: Int, .e = a: Int) => (x + y) - (z + a);

fn main() -> Int {
  return 0;
}
