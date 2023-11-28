// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}

fn Run() {
  // CHECK:STDERR: fail_empty_tuple_access.carbon:[[@LINE+3]]:3: ERROR: Index `0` is past the end of `() as type`.
  // CHECK:STDERR:   F()[0];
  // CHECK:STDERR:   ^~~~~~
  F()[0];
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc13: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_empty_tuple_access.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc13_4.1: init () = call %F.ref()
// CHECK:STDOUT:   %.loc13_7: i32 = int_literal 0
// CHECK:STDOUT:   %.loc13_4.2: ref () = temporary_storage
// CHECK:STDOUT:   %.loc13_4.3: ref () = temporary %.loc13_4.2, %.loc13_4.1
// CHECK:STDOUT:   %.loc13_8: ref <error> = tuple_index %.loc13_4.3, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
