// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "in_place_tuple_initialization.carbon" {

fn F() -> (i32, i32);
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn G() -> (i32, i32) {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-13]]_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-14]]_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-15]]_19: (type, type) = tuple_literal (%.loc[[@LINE-15]]_11, %.loc[[@LINE-15]]_16)
// CHECK:STDOUT:   %v: ref (i32, i32) = var "v"
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc[[@LINE-18]]_24: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc[[@LINE-19]]_24
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc[[@LINE-20]]: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc[[@LINE-21]]
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc[[@LINE-22]]: init (i32, i32) = call @F() to %return
// CHECK:STDOUT:   return %.loc[[@LINE-23]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
  return G()[0];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: ref (i32, i32) = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: init (i32, i32) = call @G() to %.loc[[@LINE-2]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_14: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.3: ref (i32, i32) = temporary %.loc[[@LINE-4]]_11.1, %.loc[[@LINE-4]]_11.2
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_15.1: ref i32 = tuple_index %.loc[[@LINE-5]]_11.3, %.loc[[@LINE-5]]_14
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_15.2: i32 = bind_value %.loc[[@LINE-6]]_15.1
  // CHECK:STDOUT:   return %.loc[[@LINE-7]]_15.2
  // CHECK:STDOUT: }
}
