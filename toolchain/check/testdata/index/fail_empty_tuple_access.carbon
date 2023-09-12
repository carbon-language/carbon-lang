// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_empty_tuple_access.carbon" {

fn F() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn Run() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_empty_tuple_access.carbon:[[@LINE+3]]:8: Index `0` is past the end of `() as type`.
  // CHECK:STDERR:   F()[0];
  // CHECK:STDERR:        ^
  F()[0];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_4.2: init () = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_7: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_4.3: ref () = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_4.4: ref () = temporary %.loc[[@LINE-5]]_4.3, %.loc[[@LINE-5]]_4.2
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_8: ref <error> = tuple_index %.loc[[@LINE-6]]_4.4, <error>
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
