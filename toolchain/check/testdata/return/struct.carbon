// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "struct.carbon" {

fn Main() -> {.a: i32} {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
  return {.a = 3};
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 3
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_14: i32 = stub_reference %.loc[[@LINE-2]]_16
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_17: {.a: i32} = struct_literal (%.loc[[@LINE-3]]_14)
  // CHECK:STDOUT:   return %.loc[[@LINE-4]]_17
  // CHECK:STDOUT: }
}
