// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t[0];
  var t1: i32* = &t[1];
}

// CHECK:STDOUT: file "address_of_lvalue.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_27.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc8_27.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %s: ref {.a: i32, .b: i32} = var "s"
// CHECK:STDOUT:   %.loc8_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_45: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_46.1: {.a: i32, .b: i32} = struct_literal (%.loc8_37, %.loc8_45)
// CHECK:STDOUT:   %.loc8_46.2: ref i32 = struct_access %s, member0
// CHECK:STDOUT:   %.loc8_46.3: init i32 = initialize_from %.loc8_37 to %.loc8_46.2
// CHECK:STDOUT:   %.loc8_46.4: ref i32 = struct_access %s, member1
// CHECK:STDOUT:   %.loc8_46.5: init i32 = initialize_from %.loc8_45 to %.loc8_46.4
// CHECK:STDOUT:   %.loc8_46.6: init {.a: i32, .b: i32} = struct_init %.loc8_46.1, (%.loc8_46.3, %.loc8_46.5)
// CHECK:STDOUT:   assign %s, %.loc8_46.6
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_28: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %p: ref {.a: i32, .b: i32}* = var "p"
// CHECK:STDOUT:   %s.ref.loc10: ref {.a: i32, .b: i32} = name_reference "s", %s
// CHECK:STDOUT:   %.loc10_32: {.a: i32, .b: i32}* = address_of %s.ref.loc10
// CHECK:STDOUT:   assign %p, %.loc10_32
// CHECK:STDOUT:   %.loc11_13: type = ptr_type i32
// CHECK:STDOUT:   %q: ref i32* = var "q"
// CHECK:STDOUT:   %s.ref.loc11: ref {.a: i32, .b: i32} = name_reference "s", %s
// CHECK:STDOUT:   %.loc11_19: ref i32 = struct_access %s.ref.loc11, member0
// CHECK:STDOUT:   %.loc11_17: i32* = address_of %.loc11_19
// CHECK:STDOUT:   assign %q, %.loc11_17
// CHECK:STDOUT:   %.loc12_13: type = ptr_type i32
// CHECK:STDOUT:   %r: ref i32* = var "r"
// CHECK:STDOUT:   %s.ref.loc12: ref {.a: i32, .b: i32} = name_reference "s", %s
// CHECK:STDOUT:   %.loc12_19: ref i32 = struct_access %s.ref.loc12, member1
// CHECK:STDOUT:   %.loc12_17: i32* = address_of %.loc12_19
// CHECK:STDOUT:   assign %r, %.loc12_17
// CHECK:STDOUT:   %.loc14_19.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc14_19.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc14_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc14_19.4: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %t: ref (i32, i32) = var "t"
// CHECK:STDOUT:   %.loc14_24: i32 = int_literal 1
// CHECK:STDOUT:   %.loc14_27: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_28.1: (i32, i32) = tuple_literal (%.loc14_24, %.loc14_27)
// CHECK:STDOUT:   %.loc14_28.2: ref i32 = tuple_access %t, member0
// CHECK:STDOUT:   %.loc14_28.3: init i32 = initialize_from %.loc14_24 to %.loc14_28.2
// CHECK:STDOUT:   %.loc14_28.4: ref i32 = tuple_access %t, member1
// CHECK:STDOUT:   %.loc14_28.5: init i32 = initialize_from %.loc14_27 to %.loc14_28.4
// CHECK:STDOUT:   %.loc14_28.6: init (i32, i32) = tuple_init %.loc14_28.1, (%.loc14_28.3, %.loc14_28.5)
// CHECK:STDOUT:   assign %t, %.loc14_28.6
// CHECK:STDOUT:   %.loc15_14: type = ptr_type i32
// CHECK:STDOUT:   %t0: ref i32* = var "t0"
// CHECK:STDOUT:   %t.ref.loc15: ref (i32, i32) = name_reference "t", %t
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0
// CHECK:STDOUT:   %.loc15_22: ref i32 = tuple_index %t.ref.loc15, %.loc15_21
// CHECK:STDOUT:   %.loc15_18: i32* = address_of %.loc15_22
// CHECK:STDOUT:   assign %t0, %.loc15_18
// CHECK:STDOUT:   %.loc16_14: type = ptr_type i32
// CHECK:STDOUT:   %t1: ref i32* = var "t1"
// CHECK:STDOUT:   %t.ref.loc16: ref (i32, i32) = name_reference "t", %t
// CHECK:STDOUT:   %.loc16_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc16_22: ref i32 = tuple_index %t.ref.loc16, %.loc16_21
// CHECK:STDOUT:   %.loc16_18: i32* = address_of %.loc16_22
// CHECK:STDOUT:   assign %t1, %.loc16_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
