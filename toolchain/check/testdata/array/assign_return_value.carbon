// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "assign_return_value.carbon" {

fn F() -> (i32,) { return (0,); }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn Run() {
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
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
  var t: [i32; 1] = F();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_17: type = array_type %.loc[[@LINE-2]]_16, i32
  // CHECK:STDOUT:   %t: ref [i32; 1] = var "t"
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_22.1: init (i32,) = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_22.2: ref (i32,) = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_22.3: ref (i32,) = temporary %.loc[[@LINE-6]]_22.2, %.loc[[@LINE-6]]_22.1
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_22.4: (i32,) = bind_value %.loc[[@LINE-7]]_22.3
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_22.5: [i32; 1] = array_value %.loc[[@LINE-8]]_22.4
  // CHECK:STDOUT:   assign %t, %.loc[[@LINE-9]]_22.5
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
