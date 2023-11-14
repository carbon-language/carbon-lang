// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  // CHECK:STDERR: nested_struct.carbon:[[@LINE+12]]:7: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   {.b = {.a: i32}};
  // CHECK:STDERR:       ^
  // CHECK:STDERR: nested_struct.carbon:[[@LINE+9]]:7: Member with the same name here.
  // CHECK:STDERR:   {.b = {.a: i32}};
  // CHECK:STDERR:       ^
  // CHECK:STDERR: nested_struct.carbon:[[@LINE+6]]:11: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   {.b = {.a: i32}};
  // CHECK:STDERR:           ^
  // CHECK:STDERR: nested_struct.carbon:[[@LINE+3]]:11: Member with the same name here.
  // CHECK:STDERR:   {.b = {.a: i32}};
  // CHECK:STDERR:           ^
  {.b = {.a: i32}};
  return 0;
}
