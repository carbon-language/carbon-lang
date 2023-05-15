// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/var/local/fail_missing_intro.carbon:[[@LINE+1]]: syntax error, unexpected COLON
  x: i32;
  return 1;
}
