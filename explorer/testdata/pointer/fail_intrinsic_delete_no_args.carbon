// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/pointer/fail_intrinsic_delete_no_args.carbon:[[@LINE+1]]: __intrinsic_delete takes 1 argument
  __intrinsic_delete();
  return 0;
}
