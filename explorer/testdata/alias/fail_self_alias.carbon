// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/alias/fail_self_alias.carbon:[[@LINE+1]]: 'a' is not usable until after it has been completely declared
alias a = a;

fn Main() -> i32 {
  return 0;
}
