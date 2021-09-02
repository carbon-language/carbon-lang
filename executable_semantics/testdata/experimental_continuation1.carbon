// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test that creating a continuation doesn't do anything.

fn main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
  }
  return x;
}
