// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/basic_syntax/fail_assign_to_rval.carbon:[[@LINE+1]]: Only a reference expression can be assigned to, but got `1`
  1 = 0;
  return 0;
}
