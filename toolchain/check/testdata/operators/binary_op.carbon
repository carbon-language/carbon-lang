// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  return 12 + 34;
}

// CHECK:STDOUT: file "binary_op.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_10: i32 = int_literal 12
// CHECK:STDOUT:   %.loc8_15: i32 = int_literal 34
// CHECK:STDOUT:   %.loc8_13: i32 = add %.loc8_10, %.loc8_15
// CHECK:STDOUT:   return %.loc8_13
// CHECK:STDOUT: }
