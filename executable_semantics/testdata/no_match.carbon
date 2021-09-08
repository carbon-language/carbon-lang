// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

fn main() -> i32 {
  var x: i32 = 0;
  match (x) {
    case 1 =>
      return 1;
  }
  return 0;
}
