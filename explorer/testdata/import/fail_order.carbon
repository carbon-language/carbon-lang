// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  return 0;
}

// CHECK:STDERR: SYNTAX ERROR: explorer/testdata/import/fail_order.carbon:[[@LINE+1]]: syntax error, unexpected IMPORT, expecting END_OF_FILE
import ExplorerTest library "Nonexistent";
