// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: struct.carbon:[[@LINE+6]]:16: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: fn Main() -> {.a: i32} {
// CHECK:STDERR:                ^
// CHECK:STDERR: struct.carbon:[[@LINE+3]]:16: Member with the same name here.
// CHECK:STDERR: fn Main() -> {.a: i32} {
// CHECK:STDERR:                ^
fn Main() -> {.a: i32} {
  // CHECK:STDERR: struct.carbon:[[@LINE+6]]:14: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   return {.a = 3};
  // CHECK:STDERR:              ^
  // CHECK:STDERR: struct.carbon:[[@LINE+3]]:14: Member with the same name here.
  // CHECK:STDERR:   return {.a = 3};
  // CHECK:STDERR:              ^
  return {.a = 3};
}

// CHECK:STDOUT: file "struct.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_17.1: {.a: i32} = struct_literal (%.loc8_16)
// CHECK:STDOUT:   %.loc8_17.2: {.a: i32} = struct_value (%.loc8_16)
// CHECK:STDOUT:   %.loc8_17.3: {.a: i32} = converted %.loc8_17.1, %.loc8_17.2
// CHECK:STDOUT:   return %.loc8_17.3
// CHECK:STDOUT: }
