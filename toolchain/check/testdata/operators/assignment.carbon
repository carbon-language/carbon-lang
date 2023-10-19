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
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a: ref i32 = var "a"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 12
// CHECK:STDOUT:   assign %a, %.loc8
// CHECK:STDOUT:   %a.ref.loc9: ref i32 = name_reference "a", %a
// CHECK:STDOUT:   %.loc9: i32 = int_literal 9
// CHECK:STDOUT:   assign %a.ref.loc9, %.loc9
// CHECK:STDOUT:   %.loc11_19.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc11_19.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc11_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc11_19.4: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %b: ref (i32, i32) = var "b"
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 1
// CHECK:STDOUT:   %.loc11_27: i32 = int_literal 2
// CHECK:STDOUT:   %.loc11_28.1: (i32, i32) = tuple_literal (%.loc11_24, %.loc11_27)
// CHECK:STDOUT:   %.loc11_28.2: ref i32 = tuple_access %b, member0
// CHECK:STDOUT:   %.loc11_28.3: init i32 = initialize_from %.loc11_24 to %.loc11_28.2
// CHECK:STDOUT:   %.loc11_28.4: ref i32 = tuple_access %b, member1
// CHECK:STDOUT:   %.loc11_28.5: init i32 = initialize_from %.loc11_27 to %.loc11_28.4
// CHECK:STDOUT:   %.loc11_28.6: init (i32, i32) = tuple_init %.loc11_28.1, (%.loc11_28.3, %.loc11_28.5)
// CHECK:STDOUT:   assign %b, %.loc11_28.6
// CHECK:STDOUT:   %b.ref.loc12: ref (i32, i32) = name_reference "b", %b
// CHECK:STDOUT:   %.loc12_5: i32 = int_literal 0
// CHECK:STDOUT:   %.loc12_6: ref i32 = tuple_index %b.ref.loc12, %.loc12_5
// CHECK:STDOUT:   %.loc12_10: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc12_6, %.loc12_10
// CHECK:STDOUT:   %b.ref.loc13: ref (i32, i32) = name_reference "b", %b
// CHECK:STDOUT:   %.loc13_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_6: ref i32 = tuple_index %b.ref.loc13, %.loc13_5
// CHECK:STDOUT:   %.loc13_10: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc13_6, %.loc13_10
// CHECK:STDOUT:   %.loc15_27.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc15_27.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %c: ref {.a: i32, .b: i32} = var "c"
// CHECK:STDOUT:   %.loc15_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc15_45: i32 = int_literal 2
// CHECK:STDOUT:   %.loc15_46.1: {.a: i32, .b: i32} = struct_literal (%.loc15_37, %.loc15_45)
// CHECK:STDOUT:   %.loc15_46.2: ref i32 = struct_access %c, member0
// CHECK:STDOUT:   %.loc15_46.3: init i32 = initialize_from %.loc15_37 to %.loc15_46.2
// CHECK:STDOUT:   %.loc15_46.4: ref i32 = struct_access %c, member1
// CHECK:STDOUT:   %.loc15_46.5: init i32 = initialize_from %.loc15_45 to %.loc15_46.4
// CHECK:STDOUT:   %.loc15_46.6: init {.a: i32, .b: i32} = struct_init %.loc15_46.1, (%.loc15_46.3, %.loc15_46.5)
// CHECK:STDOUT:   assign %c, %.loc15_46.6
// CHECK:STDOUT:   %c.ref.loc16: ref {.a: i32, .b: i32} = name_reference "c", %c
// CHECK:STDOUT:   %.loc16_4: ref i32 = struct_access %c.ref.loc16, member0
// CHECK:STDOUT:   %.loc16_9: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc16_4, %.loc16_9
// CHECK:STDOUT:   %c.ref.loc17: ref {.a: i32, .b: i32} = name_reference "c", %c
// CHECK:STDOUT:   %.loc17_4: ref i32 = struct_access %c.ref.loc17, member1
// CHECK:STDOUT:   %.loc17_9: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_9
// CHECK:STDOUT:   %.loc19_13: type = ptr_type i32
// CHECK:STDOUT:   %p: ref i32* = var "p"
// CHECK:STDOUT:   %a.ref.loc19: ref i32 = name_reference "a", %a
// CHECK:STDOUT:   %.loc19_17: i32* = address_of %a.ref.loc19
// CHECK:STDOUT:   assign %p, %.loc19_17
// CHECK:STDOUT:   %p.ref.loc20: ref i32* = name_reference "p", %p
// CHECK:STDOUT:   %.loc20_4: i32* = bind_value %p.ref.loc20
// CHECK:STDOUT:   %.loc20_3: ref i32 = dereference %.loc20_4
// CHECK:STDOUT:   %.loc20_8: i32 = int_literal 5
// CHECK:STDOUT:   assign %.loc20_3, %.loc20_8
// CHECK:STDOUT:   %.loc22_8: bool = bool_literal true
// CHECK:STDOUT:   if %.loc22_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc22: ref i32* = name_reference "p", %p
// CHECK:STDOUT:   %.loc22_18: i32* = bind_value %p.ref.loc22
// CHECK:STDOUT:   br !if.expr.result(%.loc22_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc22: ref i32 = name_reference "a", %a
// CHECK:STDOUT:   %.loc22_25: i32* = address_of %a.ref.loc22
// CHECK:STDOUT:   br !if.expr.result(%.loc22_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc22_5: i32* = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc22_3: ref i32 = dereference %.loc22_5
// CHECK:STDOUT:   %.loc22_31: i32 = int_literal 10
// CHECK:STDOUT:   assign %.loc22_3, %.loc22_31
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
