// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "address_of_lvalue.carbon" {

fn F() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_27: type = struct_type {.a: i32, .b: i32}
  // CHECK:STDOUT:   %s: ref {.a: i32, .b: i32} = var "s"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_37: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_35: i32 = stub_reference %.loc[[@LINE-4]]_37
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_45: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_43: i32 = stub_reference %.loc[[@LINE-6]]_45
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_46: {.a: i32, .b: i32} = struct_literal (%.loc[[@LINE-7]]_35, %.loc[[@LINE-7]]_43)
  // CHECK:STDOUT:   assign %s, %.loc[[@LINE-8]]_46

  var p: {.a: i32, .b: i32}* = &s;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_27: type = struct_type {.a: i32, .b: i32}
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_28: type = ptr_type {.a: i32, .b: i32}
  // CHECK:STDOUT:   %p: ref {.a: i32, .b: i32}* = var "p"
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_32: {.a: i32, .b: i32}* = address_of %s
  // CHECK:STDOUT:   assign %p, %.loc[[@LINE-5]]_32
  var q: i32* = &s.a;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_13: type = ptr_type i32
  // CHECK:STDOUT:   %q: ref i32* = var "q"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_19: ref i32 = struct_access %s, member0
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_17: i32* = address_of %.loc[[@LINE-4]]_19
  // CHECK:STDOUT:   assign %q, %.loc[[@LINE-5]]_17
  var r: i32* = &s.b;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_13: type = ptr_type i32
  // CHECK:STDOUT:   %r: ref i32* = var "r"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_19: ref i32 = struct_access %s, member1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_17: i32* = address_of %.loc[[@LINE-4]]_19
  // CHECK:STDOUT:   assign %r, %.loc[[@LINE-5]]_17

  var t: (i32, i32) = (1, 2);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11: type = stub_reference i32
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_16: type = stub_reference i32
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_19.1: type = tuple_type (type, type)
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_19.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_11, %.loc[[@LINE-4]]_16)
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_19.3: type = tuple_type (i32, i32)
  // CHECK:STDOUT:   %t: ref (i32, i32) = var "t"
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_24.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_24.2: i32 = stub_reference %.loc[[@LINE-8]]_24.1
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_27.1: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_27.2: i32 = stub_reference %.loc[[@LINE-10]]_27.1
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_28: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_24.2, %.loc[[@LINE-11]]_27.2)
  // CHECK:STDOUT:   assign %t, %.loc[[@LINE-12]]_28
  var t0: i32* = &t[0];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_14: type = ptr_type i32
  // CHECK:STDOUT:   %t0: ref i32* = var "t0"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_21: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_22: ref i32 = tuple_index %t, %.loc[[@LINE-4]]_21
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_18: i32* = address_of %.loc[[@LINE-5]]_22
  // CHECK:STDOUT:   assign %t0, %.loc[[@LINE-6]]_18
  var t1: i32* = &t[1];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_14: type = ptr_type i32
  // CHECK:STDOUT:   %t1: ref i32* = var "t1"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_21: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_22: ref i32 = tuple_index %t, %.loc[[@LINE-4]]_21
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_18: i32* = address_of %.loc[[@LINE-5]]_22
  // CHECK:STDOUT:   assign %t1, %.loc[[@LINE-6]]_18
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
