// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32;
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/pointer/fail_intrinsic_delete_type.carbon:[[@LINE+3]]: type error in *
  // CHECK:STDERR: expected a pointer type
  // CHECK:STDERR: actual: i32
  __intrinsic_delete(x);
  return 0;
}
