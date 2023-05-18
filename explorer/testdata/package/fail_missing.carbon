// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: SYNTAX ERROR: explorer/testdata/package/fail_missing.carbon:[[@LINE+1]]: syntax error, unexpected FN, expecting PACKAGE
fn Main() -> i32 {
  return 0;
}
