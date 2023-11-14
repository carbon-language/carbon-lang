// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  // CHECK:STDERR: two_entries.carbon:[[@LINE+24]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+21]]:12: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+18]]:21: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+15]]:21: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+12]]:35: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+9]]:35: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+6]]:43: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                           ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+3]]:43: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR:                                           ^
  var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDERR: two_entries.carbon:[[@LINE+12]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var y: {.a: i32, .b: i32} = x;
  // CHECK:STDERR:            ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+9]]:12: Member with the same name here.
  // CHECK:STDERR:   var y: {.a: i32, .b: i32} = x;
  // CHECK:STDERR:            ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+6]]:21: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var y: {.a: i32, .b: i32} = x;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: two_entries.carbon:[[@LINE+3]]:21: Member with the same name here.
  // CHECK:STDERR:   var y: {.a: i32, .b: i32} = x;
  // CHECK:STDERR:                     ^
  var y: {.a: i32, .b: i32} = x;
  return 0;
}
