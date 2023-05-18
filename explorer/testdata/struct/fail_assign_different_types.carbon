// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var p: auto = {.x = 0, .y = 0};
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/struct/fail_assign_different_types.carbon:[[@LINE+1]]: could not find implementation of interface AssignWith(U = {.y: i32}) for {.x: i32, .y: i32}
  p = {.y = 0};
  return 0;
}
