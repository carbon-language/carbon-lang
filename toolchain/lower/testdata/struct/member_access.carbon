// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  // CHECK:STDERR: member_access.carbon:[[@LINE+24]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+21]]:12: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+18]]:21: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+15]]:21: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+12]]:35: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+9]]:35: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+6]]:45: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                                             ^
  // CHECK:STDERR: member_access.carbon:[[@LINE+3]]:45: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  // CHECK:STDERR:                                             ^
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}
