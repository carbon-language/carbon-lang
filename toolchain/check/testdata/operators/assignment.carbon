// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var a: i32 = 12;
  a = 9;

  var b: (i32, i32) = (1, 2);
  b[0] = 3;
  b[1] = 4;

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  c.a = 3;
  c.b = 4;

  var p: i32* = &a;
  *p = 5;

  *(if true then p else &a) = 10;
}

// CHECK:STDOUT: file "assignment.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a: ref i32 = var "a"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 12
// CHECK:STDOUT:   assign %a, %.loc8
// CHECK:STDOUT:   %.loc9: i32 = int_literal 9
// CHECK:STDOUT:   assign %a, %.loc9
// CHECK:STDOUT:   %.loc11_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc11_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc11_19.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc11_19.2: (type, type) = tuple_literal (%.loc11_11, %.loc11_16)
// CHECK:STDOUT:   %.loc11_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %b: ref (i32, i32) = var "b"
// CHECK:STDOUT:   %.loc11_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc11_24.2: i32 = stub_reference %.loc11_24.1
// CHECK:STDOUT:   %.loc11_27.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc11_27.2: i32 = stub_reference %.loc11_27.1
// CHECK:STDOUT:   %.loc11_28: (i32, i32) = tuple_literal (%.loc11_24.2, %.loc11_27.2)
// CHECK:STDOUT:   assign %b, %.loc11_28
// CHECK:STDOUT:   %.loc12_5: i32 = int_literal 0
// CHECK:STDOUT:   %.loc12_6: ref i32 = tuple_index %b, %.loc12_5
// CHECK:STDOUT:   %.loc12_10: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc12_6, %.loc12_10
// CHECK:STDOUT:   %.loc13_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_6: ref i32 = tuple_index %b, %.loc13_5
// CHECK:STDOUT:   %.loc13_10: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc13_6, %.loc13_10
// CHECK:STDOUT:   %.loc15_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %c: ref {.a: i32, .b: i32} = var "c"
// CHECK:STDOUT:   %.loc15_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc15_35: i32 = stub_reference %.loc15_37
// CHECK:STDOUT:   %.loc15_45: i32 = int_literal 2
// CHECK:STDOUT:   %.loc15_43: i32 = stub_reference %.loc15_45
// CHECK:STDOUT:   %.loc15_46: {.a: i32, .b: i32} = struct_literal (%.loc15_35, %.loc15_43)
// CHECK:STDOUT:   assign %c, %.loc15_46
// CHECK:STDOUT:   %.loc16_4: ref i32 = struct_access %c, member0
// CHECK:STDOUT:   %.loc16_9: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc16_4, %.loc16_9
// CHECK:STDOUT:   %.loc17_4: ref i32 = struct_access %c, member1
// CHECK:STDOUT:   %.loc17_9: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_9
// CHECK:STDOUT:   %.loc19_13: type = ptr_type i32
// CHECK:STDOUT:   %p: ref i32* = var "p"
// CHECK:STDOUT:   %.loc19_17: i32* = address_of %a
// CHECK:STDOUT:   assign %p, %.loc19_17
// CHECK:STDOUT:   %.loc19_7.1: i32* = bind_value %p
// CHECK:STDOUT:   %.loc20_3: ref i32 = dereference %.loc19_7.1
// CHECK:STDOUT:   %.loc20_8: i32 = int_literal 5
// CHECK:STDOUT:   assign %.loc20_3, %.loc20_8
// CHECK:STDOUT:   %.loc22_8: bool = bool_literal true
// CHECK:STDOUT:   if %.loc22_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc19_7.2: i32* = bind_value %p
// CHECK:STDOUT:   br !if.expr.result(%.loc19_7.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc22_25: i32* = address_of %a
// CHECK:STDOUT:   br !if.expr.result(%.loc22_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc22_5: i32* = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc22_3: ref i32 = dereference %.loc22_5
// CHECK:STDOUT:   %.loc22_31: i32 = int_literal 10
// CHECK:STDOUT:   assign %.loc22_3, %.loc22_31
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
