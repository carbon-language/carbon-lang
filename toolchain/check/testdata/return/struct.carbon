// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: file "struct.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_14: i32 = stub_reference %.loc8_16
// CHECK:STDOUT:   %.loc8_17: {.a: i32} = struct_literal (%.loc8_14)
// CHECK:STDOUT:   return %.loc8_17
// CHECK:STDOUT: }
