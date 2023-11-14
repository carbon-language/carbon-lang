// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var data: i32;
  var next: C*;
}

fn Make() -> C {
  returned var c: C;
  // CHECK:STDERR: return_var.carbon:[[@LINE+12]]:14: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   c = {.data = 42, .next = &c};
  // CHECK:STDERR:              ^
  // CHECK:STDERR: return_var.carbon:[[@LINE+9]]:14: Member with the same name here.
  // CHECK:STDERR:   c = {.data = 42, .next = &c};
  // CHECK:STDERR:              ^
  // CHECK:STDERR: return_var.carbon:[[@LINE+6]]:26: ERROR: Member of struct literal with a duplicated name.
  // CHECK:STDERR:   c = {.data = 42, .next = &c};
  // CHECK:STDERR:                          ^
  // CHECK:STDERR: return_var.carbon:[[@LINE+3]]:26: Member with the same name here.
  // CHECK:STDERR:   c = {.data = 42, .next = &c};
  // CHECK:STDERR:                          ^
  c = {.data = 42, .next = &c};
  return var;
}
