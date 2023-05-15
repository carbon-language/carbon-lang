// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32;
  if (0 == 0) {
    x = 0;
  }
  // Static analysis thinks `x` may be formed, defer the check to run-time.
  return x;
}
