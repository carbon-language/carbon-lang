// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/basic_syntax/fail_unknown_intrinsic.carbon:[[@LINE+1]]: Unknown intrinsic 'nonexistent'
  return __intrinsic_nonexistent();
}
