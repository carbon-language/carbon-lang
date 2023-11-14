// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  // CHECK:STDERR: one_entry.carbon:[[@LINE+12]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32} = {.a = 4};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: one_entry.carbon:[[@LINE+9]]:12: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32} = {.a = 4};
  // CHECK:STDERR:            ^
  // CHECK:STDERR: one_entry.carbon:[[@LINE+6]]:26: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var x: {.a: i32} = {.a = 4};
  // CHECK:STDERR:                          ^
  // CHECK:STDERR: one_entry.carbon:[[@LINE+3]]:26: Member with the same name here.
  // CHECK:STDERR:   var x: {.a: i32} = {.a = 4};
  // CHECK:STDERR:                          ^
  var x: {.a: i32} = {.a = 4};
  // CHECK:STDERR: one_entry.carbon:[[@LINE+6]]:12: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   var y: {.a: i32} = x;
  // CHECK:STDERR:            ^
  // CHECK:STDERR: one_entry.carbon:[[@LINE+3]]:12: Member with the same name here.
  // CHECK:STDERR:   var y: {.a: i32} = x;
  // CHECK:STDERR:            ^
  var y: {.a: i32} = x;
  return 0;
}
