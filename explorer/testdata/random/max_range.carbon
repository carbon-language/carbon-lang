// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  // -2147483648 is unsupported as a literal, but we can do it with a decrement.
  var low: i32 = -2147483647;
  low -= 1;
  var high: i32 = 2147483647;
  Rand(low, high);
  return 0;
}
