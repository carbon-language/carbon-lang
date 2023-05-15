// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// CHECK:STDERR: SYNTAX ERROR: explorer/testdata/var/global/fail_type_only.carbon:[[@LINE+1]]: syntax error, unexpected sized_type_literal, expecting identifier
var i32;

fn Main() -> i32 {
  return 0;
}
