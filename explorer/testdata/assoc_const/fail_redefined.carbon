// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Iface {
  let T:! type;
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/assoc_const/fail_redefined.carbon:[[@LINE+1]]: Duplicate name `T` also found at explorer/testdata/assoc_const/fail_redefined.carbon:[[@LINE-1]]
  let T:! type;
}

fn Main() -> i32 { return 0; }
