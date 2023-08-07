// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
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
// CHECK:STDOUT:   node+18,
// CHECK:STDOUT:   node+23,
// CHECK:STDOUT:   node+37,
// CHECK:STDOUT:   node+39,
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
// CHECK:STDOUT:   {kind: Assign, arg0: node+6, arg1: node+14},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str5, arg1: node+19, type: type2},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+6, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+19, arg1: node+21},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str6, arg1: node+24, type: type3},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+6, arg1: member0, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+26, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+24, arg1: node+27},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str7, arg1: node+30, type: type3},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+6, arg1: member1, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+32, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+30, arg1: node+33},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block9, type: type4},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type5},
// CHECK:STDOUT:   {kind: BindName, arg0: str8, arg1: node+40, type: type5},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+42, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+44, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block10, type: type5},
// CHECK:STDOUT:   {kind: Assign, arg0: node+40, arg1: node+46},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str9, arg1: node+49, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: Index, arg0: node+40, arg1: node+51, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+52, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+49, arg1: node+53},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str10, arg1: node+56, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: Index, arg0: node+40, arg1: node+58, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+59, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+56, arg1: node+60},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str11, arg1: node+63, type: type3},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+0, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+63, arg1: node+65},
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
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+43,
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

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
