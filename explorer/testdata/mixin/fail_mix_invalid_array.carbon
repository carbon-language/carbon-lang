// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/mixin/fail_mix_invalid_array.carbon:[[@LINE+1]]: type error in type expression: 'bool' is not implicitly convertible to 'type'
  __mix [true; true];
}

fn Main() -> i32 {
  return 0;
}
