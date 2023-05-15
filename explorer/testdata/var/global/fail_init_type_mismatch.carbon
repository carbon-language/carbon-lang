// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// Test type checking of global variable. Error expected.

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/global/fail_init_type_mismatch.carbon:[[@LINE+1]]: type error in initializer of variable: 'bool' is not implicitly convertible to 'i32'
var flag: i32 = true;

fn Main() -> i32 {
  return 0;
}
