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
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %s: ref {.a: i32, .b: i32} = var "s"
// CHECK:STDOUT:   %.loc8_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_35: i32 = stub_reference %.loc8_37
// CHECK:STDOUT:   %.loc8_45: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_43: i32 = stub_reference %.loc8_45
// CHECK:STDOUT:   %.loc8_46: {.a: i32, .b: i32} = struct_literal (%.loc8_35, %.loc8_43)
// CHECK:STDOUT:   assign %s, %.loc8_46
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_28: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %p: ref {.a: i32, .b: i32}* = var "p"
// CHECK:STDOUT:   %.loc10_32: {.a: i32, .b: i32}* = address_of %s
// CHECK:STDOUT:   assign %p, %.loc10_32
// CHECK:STDOUT:   %.loc11_13: type = ptr_type i32
// CHECK:STDOUT:   %q: ref i32* = var "q"
// CHECK:STDOUT:   %.loc11_19: ref i32 = struct_access %s, member0
// CHECK:STDOUT:   %.loc11_17: i32* = address_of %.loc11_19
// CHECK:STDOUT:   assign %q, %.loc11_17
// CHECK:STDOUT:   %.loc12_13: type = ptr_type i32
// CHECK:STDOUT:   %r: ref i32* = var "r"
// CHECK:STDOUT:   %.loc12_19: ref i32 = struct_access %s, member1
// CHECK:STDOUT:   %.loc12_17: i32* = address_of %.loc12_19
// CHECK:STDOUT:   assign %r, %.loc12_17
// CHECK:STDOUT:   %.loc14_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc14_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc14_19.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc14_19.2: (type, type) = tuple_literal (%.loc14_11, %.loc14_16)
// CHECK:STDOUT:   %.loc14_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %t: ref (i32, i32) = var "t"
// CHECK:STDOUT:   %.loc14_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc14_24.2: i32 = stub_reference %.loc14_24.1
// CHECK:STDOUT:   %.loc14_27.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_27.2: i32 = stub_reference %.loc14_27.1
// CHECK:STDOUT:   %.loc14_28: (i32, i32) = tuple_literal (%.loc14_24.2, %.loc14_27.2)
// CHECK:STDOUT:   assign %t, %.loc14_28
// CHECK:STDOUT:   %.loc15_14: type = ptr_type i32
// CHECK:STDOUT:   %t0: ref i32* = var "t0"
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0
// CHECK:STDOUT:   %.loc15_22: ref i32 = tuple_index %t, %.loc15_21
// CHECK:STDOUT:   %.loc15_18: i32* = address_of %.loc15_22
// CHECK:STDOUT:   assign %t0, %.loc15_18
// CHECK:STDOUT:   %.loc16_14: type = ptr_type i32
// CHECK:STDOUT:   %t1: ref i32* = var "t1"
// CHECK:STDOUT:   %.loc16_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc16_22: ref i32 = tuple_index %t, %.loc16_21
// CHECK:STDOUT:   %.loc16_18: i32* = address_of %.loc16_22
// CHECK:STDOUT:   assign %t1, %.loc16_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
