// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: struct_param.carbon:[[@LINE+18]]:18: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                  ^
// CHECK:STDERR: struct_param.carbon:[[@LINE+15]]:18: Member with the same name here.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                  ^
// CHECK:STDERR: struct_param.carbon:[[@LINE+12]]:32: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                                ^
// CHECK:STDERR: struct_param.carbon:[[@LINE+9]]:32: Member with the same name here.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                                ^
// CHECK:STDERR: struct_param.carbon:[[@LINE+6]]:41: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                                         ^
// CHECK:STDERR: struct_param.carbon:[[@LINE+3]]:41: Member with the same name here.
// CHECK:STDERR: fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}
// CHECK:STDERR:                                         ^
fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}

fn Main() {
  // CHECK:STDERR: struct_param.carbon:[[@LINE+18]]:13: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:             ^
  // CHECK:STDERR: struct_param.carbon:[[@LINE+15]]:13: Member with the same name here.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:             ^
  // CHECK:STDERR: struct_param.carbon:[[@LINE+12]]:23: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: struct_param.carbon:[[@LINE+9]]:23: Member with the same name here.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: struct_param.carbon:[[@LINE+6]]:31: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: struct_param.carbon:[[@LINE+3]]:31: Member with the same name here.
  // CHECK:STDERR:   F({}, {.a = 1}, {.a = 2, .b = 3});
  // CHECK:STDERR:                               ^
  F({}, {.a = 1}, {.a = 2, .b = 3});
}
