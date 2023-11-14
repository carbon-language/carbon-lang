// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*);

fn F() {
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+24]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+21]]:12: Member with the same name here.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+18]]:21: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+15]]:21: Member with the same name here.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+12]]:35: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+9]]:35: Member with the same name here.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+6]]:43: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                           ^
  // CHECK:STDERR: address_of_field.carbon:[[@LINE+3]]:43: Member with the same name here.
  // CHECK:STDERR:   var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                           ^
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(&s.b);
}
