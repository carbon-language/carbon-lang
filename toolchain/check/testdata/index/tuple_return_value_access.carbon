// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "tuple_return_value_access.carbon" {

fn F() -> (i32,) { return (0,); }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn Run() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-9]]_28.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc[[@LINE-10]]_28.2: i32 = stub_reference %.loc[[@LINE-10]]_28.1
// CHECK:STDOUT:   %.loc[[@LINE-11]]_30: (i32,) = tuple_literal (%.loc[[@LINE-11]]_28.2)
// CHECK:STDOUT:   return %.loc[[@LINE-12]]_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
  return F()[0];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: init (i32,) = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_14: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_11.2: ref (i32,) = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.3: ref (i32,) = temporary %.loc[[@LINE-4]]_11.2, %.loc[[@LINE-4]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_15.1: ref i32 = tuple_index %.loc[[@LINE-5]]_11.3, %.loc[[@LINE-5]]_14
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_15.2: i32 = bind_value %.loc[[@LINE-6]]_15.1
  // CHECK:STDOUT:   return %.loc[[@LINE-7]]_15.2
  // CHECK:STDOUT: }
}
