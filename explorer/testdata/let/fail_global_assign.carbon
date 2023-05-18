// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

let x: i32 = 10;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/let/fail_global_assign.carbon:[[@LINE+1]]: Only a reference expression can be assigned to, but got `x`
  x = 0;
  return 0;
}
