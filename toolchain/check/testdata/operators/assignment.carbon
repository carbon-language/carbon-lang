// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "assignment.carbon" {

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a: ref i32 = var "a"
  var a: i32 = 12;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 12
  // CHECK:STDOUT:   assign %a, %.loc[[@LINE-2]]
  a = 9;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 9
  // CHECK:STDOUT:   assign %a, %.loc[[@LINE-2]]

  var b: (i32, i32) = (1, 2);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11: type = stub_reference i32
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_16: type = stub_reference i32
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_19.1: type = tuple_type (type, type)
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_19.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_11, %.loc[[@LINE-4]]_16)
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_19.3: type = tuple_type (i32, i32)
  // CHECK:STDOUT:   %b: ref (i32, i32) = var "b"
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_24.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_24.2: i32 = stub_reference %.loc[[@LINE-8]]_24.1
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_27.1: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_27.2: i32 = stub_reference %.loc[[@LINE-10]]_27.1
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_28: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_24.2, %.loc[[@LINE-11]]_27.2)
  // CHECK:STDOUT:   assign %b, %.loc[[@LINE-12]]_28
  b[0] = 3;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_5: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_6: ref i32 = tuple_index %b, %.loc[[@LINE-2]]_5
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_10: i32 = int_literal 3
  // CHECK:STDOUT:   assign %.loc[[@LINE-4]]_6, %.loc[[@LINE-4]]_10
  b[1] = 4;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_5: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_6: ref i32 = tuple_index %b, %.loc[[@LINE-2]]_5
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_10: i32 = int_literal 4
  // CHECK:STDOUT:   assign %.loc[[@LINE-4]]_6, %.loc[[@LINE-4]]_10

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_27: type = struct_type {.a: i32, .b: i32}
  // CHECK:STDOUT:   %c: ref {.a: i32, .b: i32} = var "c"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_37: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_35: i32 = stub_reference %.loc[[@LINE-4]]_37
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_45: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_43: i32 = stub_reference %.loc[[@LINE-6]]_45
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_46: {.a: i32, .b: i32} = struct_literal (%.loc[[@LINE-7]]_35, %.loc[[@LINE-7]]_43)
  // CHECK:STDOUT:   assign %c, %.loc[[@LINE-8]]_46
  c.a = 3;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4: ref i32 = struct_access %c, member0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_9: i32 = int_literal 3
  // CHECK:STDOUT:   assign %.loc[[@LINE-3]]_4, %.loc[[@LINE-3]]_9
  c.b = 4;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4: ref i32 = struct_access %c, member1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_9: i32 = int_literal 4
  // CHECK:STDOUT:   assign %.loc[[@LINE-3]]_4, %.loc[[@LINE-3]]_9

  var p: i32* = &a;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_13: type = ptr_type i32
  // CHECK:STDOUT:   %p: ref i32* = var "p"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_17: i32* = address_of %a
  // CHECK:STDOUT:   assign %p, %.loc[[@LINE-4]]_17
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_7.1: i32* = bind_value %p
  *p = 5;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_3: ref i32 = dereference %.loc[[@LINE-7]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_8: i32 = int_literal 5
  // CHECK:STDOUT:   assign %.loc[[@LINE-3]]_3, %.loc[[@LINE-3]]_8

  *(if true then p else &a) = 10;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_8: bool = bool_literal true
  // CHECK:STDOUT:   if %.loc[[@LINE-2]]_8 br !if.expr.then else br !if.expr.else
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then:
  // CHECK:STDOUT:   %.loc[[@LINE-16]]_7.2: i32* = bind_value %p
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-17]]_7.2)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else:
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_25: i32* = address_of %a
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-10]]_25)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result:
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_5: i32* = block_arg !if.expr.result
  // CHECK:STDOUT:   %.loc[[@LINE-14]]_3: ref i32 = dereference %.loc[[@LINE-14]]_5
  // CHECK:STDOUT:   %.loc[[@LINE-15]]_31: i32 = int_literal 10
  // CHECK:STDOUT:   assign %.loc[[@LINE-16]]_3, %.loc[[@LINE-16]]_31
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
