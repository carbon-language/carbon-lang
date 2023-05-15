// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/array/fail_size_mismatch.carbon:[[@LINE+1]]: type error in initializer of variable: '(i32, i32, i32)' is not implicitly convertible to '[i32; 2]'
  var x: [i32; 2] = (0, 1, 2);
  return x[0];
}
