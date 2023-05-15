// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/operators/fail_overflow_div.carbon:[[@LINE+1]]: integer overflow
  var a: auto = (-2147483647 -1) /-1;
  return 0;
}
