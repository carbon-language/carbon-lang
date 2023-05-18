// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 5;
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/pointer/fail_rvalue_addressof.carbon:[[@LINE+1]]: Argument to & should be a reference expression.
  var p: i32* = &5;
  return 0;
}
