// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var x: auto = (0, 1);
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/tuple/fail_index.carbon:[[@LINE+1]]: index 2 is out of range for type (i32, i32)
  return x[2];
}
