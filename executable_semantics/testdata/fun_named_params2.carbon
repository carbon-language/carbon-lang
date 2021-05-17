// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(Int x, .d = Int y, Int z, .e = Int a) => (x + y) - (z + a);

fn main() -> Int {
  return 0;
}
