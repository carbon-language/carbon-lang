// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/print/fail_intrinsic_type.carbon:[[@LINE+2]]: type error in Print argument 1
  // CHECK:STDERR: expected:
  __intrinsic_print("{0}", {.x = 1});
  return 0;
}
