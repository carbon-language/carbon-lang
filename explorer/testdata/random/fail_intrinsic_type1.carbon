// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/random/fail_intrinsic_type1.carbon:[[@LINE+3]]: type error in Rand argument 1
  // CHECK:STDERR: expected: i32
  // CHECK:STDERR: actual: String
  __intrinsic_rand(1, "a");
  return 0;
}
