// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/random/fail_intrinsic_no_args.carbon:[[@LINE+1]]: Rand takes 2 arguments, received 0
  __intrinsic_rand();
  return 0;
}
