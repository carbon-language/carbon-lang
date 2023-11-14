// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+24]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+21]]:12: Member with the same name here.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+18]]:33: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+15]]:33: Member with the same name here.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+12]]:59: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                                           ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+9]]:59: Member with the same name here.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                                           ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+6]]:69: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                                                     ^
  // CHECK:STDERR: nested_struct_in_place.carbon:[[@LINE+3]]:69: Member with the same name here.
  // CHECK:STDERR:   var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
  // CHECK:STDERR:                                                                     ^
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}
