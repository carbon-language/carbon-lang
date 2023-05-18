// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Iface {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/assoc_const/fail_anonymous.carbon:[[@LINE+1]]: syntax error, unexpected UNDERSCORE, expecting identifier or TEMPLATE
  let _:! type;
}

fn Main() -> i32 { return 0; }
