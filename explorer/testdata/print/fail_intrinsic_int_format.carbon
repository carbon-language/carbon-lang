// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/print/fail_intrinsic_int_format.carbon:[[@LINE+3]]: type error in Print argument 0
  // CHECK:STDERR: expected: String
  // CHECK:STDERR: actual: i32
  __intrinsic_print(0);
}
