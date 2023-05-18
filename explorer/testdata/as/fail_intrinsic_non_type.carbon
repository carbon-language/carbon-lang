// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/as/fail_intrinsic_non_type.carbon:[[@LINE+1]]: type error in type expression: 'i32' is not implicitly convertible to 'type'
  __intrinsic_implicit_as(0);
  return 0;
}
