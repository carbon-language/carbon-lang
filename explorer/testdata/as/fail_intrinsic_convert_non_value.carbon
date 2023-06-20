// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() {}

fn Main() -> i32 {
  // Validate that this doesn't crash.
  // CHECK:STDERR: COMPILATION ERROR: fail_intrinsic_convert_non_value.carbon:[[@LINE+1]]: no builtin conversion from fn () -> () to i32 is known
  __intrinsic_implicit_as_convert(F, i32);
  return 0;
}
