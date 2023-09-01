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

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: [block2]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   s,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   p,
// CHECK:STDOUT:   q,
// CHECK:STDOUT:   r,
// CHECK:STDOUT:   t,
// CHECK:STDOUT:   t0,
// CHECK:STDOUT:   t1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+16,
// CHECK:STDOUT:   node+20,
// CHECK:STDOUT:   node+32,
// CHECK:STDOUT:   node+34,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block3, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block5, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+4, arg1: node+11},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block6, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str4, type: type2},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+4, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+17, arg1: node+18},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str5, type: type3},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+4, arg1: member0, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+22, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+21, arg1: node+23},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str6, type: type3},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+4, arg1: member1, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+27, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+26, arg1: node+28},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block7, type: type4},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str7, type: type5},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+36, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+38, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block8, type: type5},
// CHECK:STDOUT:   {kind: Assign, arg0: node+35, arg1: node+40},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str8, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+35, arg1: node+44, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+45, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+43, arg1: node+46},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str9, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+35, arg1: node+50, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+51, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+49, arg1: node+52},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+33,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
// CHECK:STDOUT:     node+42,
// CHECK:STDOUT:     node+43,
// CHECK:STDOUT:     node+44,
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:     node+46,
// CHECK:STDOUT:     node+47,
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:     node+49,
// CHECK:STDOUT:     node+50,
// CHECK:STDOUT:     node+51,
// CHECK:STDOUT:     node+52,
// CHECK:STDOUT:     node+53,
// CHECK:STDOUT:     node+54,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
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
// CHECK:STDOUT:   %.loc8_46: {.a: i32, .b: i32} = struct_value (%.loc8_35, %.loc8_43)
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
// CHECK:STDOUT:   %.loc14_19.2: (type, type) = tuple_value (%.loc14_11, %.loc14_16)
// CHECK:STDOUT:   %.loc14_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %t: ref (i32, i32) = var "t"
// CHECK:STDOUT:   %.loc14_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc14_24.2: i32 = stub_reference %.loc14_24.1
// CHECK:STDOUT:   %.loc14_27.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_27.2: i32 = stub_reference %.loc14_27.1
// CHECK:STDOUT:   %.loc14_28: (i32, i32) = tuple_value (%.loc14_24.2, %.loc14_27.2)
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
