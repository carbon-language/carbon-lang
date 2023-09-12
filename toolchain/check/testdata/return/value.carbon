// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "value.carbon" {

fn Main() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
  return 0;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 0
  // CHECK:STDOUT:   return %.loc[[@LINE-2]]
  // CHECK:STDOUT: }
}
