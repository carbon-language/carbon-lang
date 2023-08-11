// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(param: i32) {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t[0];
  var t1: i32* = &t[1];

  // TODO: Once we start treating parameters as `let` bindings by default, we
  // should reject this, but `fn F(var n: i32)` should still work.
  var param_addr: i32* = &param;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, body: {block4}}},
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
// CHECK:STDOUT:   param,
// CHECK:STDOUT:   s,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   p,
// CHECK:STDOUT:   q,
// CHECK:STDOUT:   r,
// CHECK:STDOUT:   t,
// CHECK:STDOUT:   t0,
// CHECK:STDOUT:   t1,
// CHECK:STDOUT:   param_addr,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   node+20,
// CHECK:STDOUT:   node+26,
// CHECK:STDOUT:   node+42,
// CHECK:STDOUT:   node+44,
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
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block5, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+6, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+11, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block7, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+14, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+6, arg1: node+15},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block8, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str5, arg1: node+21, type: type2},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+6, type: type2},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+23, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+21, arg1: node+24},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str6, arg1: node+27, type: type3},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+6, arg1: member0, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+29, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+30, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+27, arg1: node+31},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str7, arg1: node+34, type: type3},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+6, arg1: member1, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+36, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+37, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+34, arg1: node+38},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block9, type: type4},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type5},
// CHECK:STDOUT:   {kind: BindName, arg0: str8, arg1: node+45, type: type5},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+47, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+49, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block10, type: type5},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+51, type: type5},
// CHECK:STDOUT:   {kind: Assign, arg0: node+45, arg1: node+52},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str9, arg1: node+55, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: Index, arg0: node+45, arg1: node+57, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+58, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+59, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+55, arg1: node+60},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str10, arg1: node+63, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: Index, arg0: node+45, arg1: node+65, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+66, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+67, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+63, arg1: node+68},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str11, arg1: node+71, type: type3},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+0, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+73, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+71, arg1: node+74},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
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
// CHECK:STDOUT:     node+55,
// CHECK:STDOUT:     node+56,
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:     node+58,
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:     node+60,
// CHECK:STDOUT:     node+61,
// CHECK:STDOUT:     node+62,
// CHECK:STDOUT:     node+63,
// CHECK:STDOUT:     node+64,
// CHECK:STDOUT:     node+65,
// CHECK:STDOUT:     node+66,
// CHECK:STDOUT:     node+67,
// CHECK:STDOUT:     node+68,
// CHECK:STDOUT:     node+69,
// CHECK:STDOUT:     node+70,
// CHECK:STDOUT:     node+71,
// CHECK:STDOUT:     node+72,
// CHECK:STDOUT:     node+73,
// CHECK:STDOUT:     node+74,
// CHECK:STDOUT:     node+75,
// CHECK:STDOUT:     node+76,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:     node+50,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%param: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %s: {.a: i32, .b: i32} = var
// CHECK:STDOUT:   %.loc8_37: i32 = int_value 1
// CHECK:STDOUT:   %.loc8_35: i32 = stub_reference %.loc8_37
// CHECK:STDOUT:   %.loc8_45: i32 = int_value 2
// CHECK:STDOUT:   %.loc8_43: i32 = stub_reference %.loc8_45
// CHECK:STDOUT:   %.loc8_46.1: {.a: i32, .b: i32} = struct_value (%.loc8_35, %.loc8_43)
// CHECK:STDOUT:   %.loc8_46.2: {.a: i32, .b: i32} = initialize_from %.loc8_46.1
// CHECK:STDOUT:   assign %s, %.loc8_46.2
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_28: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %p: {.a: i32, .b: i32}* = var
// CHECK:STDOUT:   %.loc10_32.1: {.a: i32, .b: i32}* = address_of %s
// CHECK:STDOUT:   %.loc10_32.2: {.a: i32, .b: i32}* = initialize_from %.loc10_32.1
// CHECK:STDOUT:   assign %p, %.loc10_32.2
// CHECK:STDOUT:   %.loc11_13: type = ptr_type i32
// CHECK:STDOUT:   %q: i32* = var
// CHECK:STDOUT:   %.loc11_19: i32 = struct_access %s, member0
// CHECK:STDOUT:   %.loc11_17.1: i32* = address_of %.loc11_19
// CHECK:STDOUT:   %.loc11_17.2: i32* = initialize_from %.loc11_17.1
// CHECK:STDOUT:   assign %q, %.loc11_17.2
// CHECK:STDOUT:   %.loc12_13: type = ptr_type i32
// CHECK:STDOUT:   %r: i32* = var
// CHECK:STDOUT:   %.loc12_19: i32 = struct_access %s, member1
// CHECK:STDOUT:   %.loc12_17.1: i32* = address_of %.loc12_19
// CHECK:STDOUT:   %.loc12_17.2: i32* = initialize_from %.loc12_17.1
// CHECK:STDOUT:   assign %r, %.loc12_17.2
// CHECK:STDOUT:   %.loc14_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc14_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc14_19.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc14_19.2: (type, type) = tuple_value (%.loc14_11, %.loc14_16)
// CHECK:STDOUT:   %.loc14_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %t: (i32, i32) = var
// CHECK:STDOUT:   %.loc14_24.1: i32 = int_value 1
// CHECK:STDOUT:   %.loc14_24.2: i32 = stub_reference %.loc14_24.1
// CHECK:STDOUT:   %.loc14_27.1: i32 = int_value 2
// CHECK:STDOUT:   %.loc14_27.2: i32 = stub_reference %.loc14_27.1
// CHECK:STDOUT:   %.loc14_28.1: (i32, i32) = tuple_value (%.loc14_24.2, %.loc14_27.2)
// CHECK:STDOUT:   %.loc14_28.2: (i32, i32) = initialize_from %.loc14_28.1
// CHECK:STDOUT:   assign %t, %.loc14_28.2
// CHECK:STDOUT:   %.loc15_14: type = ptr_type i32
// CHECK:STDOUT:   %t0: i32* = var
// CHECK:STDOUT:   %.loc15_21: i32 = int_value 0
// CHECK:STDOUT:   %.loc15_22: i32 = tuple_index %t, %.loc15_21
// CHECK:STDOUT:   %.loc15_18.1: i32* = address_of %.loc15_22
// CHECK:STDOUT:   %.loc15_18.2: i32* = initialize_from %.loc15_18.1
// CHECK:STDOUT:   assign %t0, %.loc15_18.2
// CHECK:STDOUT:   %.loc16_14: type = ptr_type i32
// CHECK:STDOUT:   %t1: i32* = var
// CHECK:STDOUT:   %.loc16_21: i32 = int_value 1
// CHECK:STDOUT:   %.loc16_22: i32 = tuple_index %t, %.loc16_21
// CHECK:STDOUT:   %.loc16_18.1: i32* = address_of %.loc16_22
// CHECK:STDOUT:   %.loc16_18.2: i32* = initialize_from %.loc16_18.1
// CHECK:STDOUT:   assign %t1, %.loc16_18.2
// CHECK:STDOUT:   %.loc20_22: type = ptr_type i32
// CHECK:STDOUT:   %param_addr: i32* = var
// CHECK:STDOUT:   %.loc20_26.1: i32* = address_of %param
// CHECK:STDOUT:   %.loc20_26.2: i32* = initialize_from %.loc20_26.1
// CHECK:STDOUT:   assign %param_addr, %.loc20_26.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
