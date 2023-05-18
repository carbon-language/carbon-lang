// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/local/fail_nested_type_only.carbon:[[@LINE+1]]: An irrefutable pattern is required, but `i32` is refutable.
  var (x: i32, i32);
  return 0;
}
