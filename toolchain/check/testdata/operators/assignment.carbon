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

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: [block2, block8, block9, block10]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   12,
// CHECK:STDOUT:   -7,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   10,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Main,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT:   p,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+8,
// CHECK:STDOUT:   node+10,
// CHECK:STDOUT:   node+28,
// CHECK:STDOUT:   node+44,
// CHECK:STDOUT:   nodeBoolType,
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
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+1, arg1: node+2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+1, arg1: node+4},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type1},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str2, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+12, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+14, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+11, arg1: node+16},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+11, arg1: node+18, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+19, arg1: node+20},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int6, type: type0},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+11, arg1: node+22, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int7, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+23, arg1: node+24},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block5, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str3, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int8, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+30, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int9, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+33, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block7, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+29, arg1: node+36},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+29, arg1: member0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int10, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+38, arg1: node+39},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+29, arg1: member1, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int11, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+41, arg1: node+42},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str4, type: type4},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+1, type: type4},
// CHECK:STDOUT:   {kind: Assign, arg0: node+45, arg1: node+46},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+45, type: type4},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+48, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int12, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+49, arg1: node+50},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type5},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block8, arg1: node+52},
// CHECK:STDOUT:   {kind: Branch, arg0: block9},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+45, type: type4},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+1, type: type4},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block10, arg1: node+55},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block10, arg1: node+56},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block10, type: type4},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+59, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int13, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+60, arg1: node+61},
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
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
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
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+33,
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
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+55,
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+56,
// CHECK:STDOUT:     node+58,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:     node+60,
// CHECK:STDOUT:     node+61,
// CHECK:STDOUT:     node+62,
// CHECK:STDOUT:     node+63,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
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
// CHECK:STDOUT:   %.loc11_19.2: (type, type) = tuple_value (%.loc11_11, %.loc11_16)
// CHECK:STDOUT:   %.loc11_19.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %b: ref (i32, i32) = var "b"
// CHECK:STDOUT:   %.loc11_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc11_24.2: i32 = stub_reference %.loc11_24.1
// CHECK:STDOUT:   %.loc11_27.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc11_27.2: i32 = stub_reference %.loc11_27.1
// CHECK:STDOUT:   %.loc11_28: (i32, i32) = tuple_value (%.loc11_24.2, %.loc11_27.2)
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
// CHECK:STDOUT:   %.loc15_46: {.a: i32, .b: i32} = struct_value (%.loc15_35, %.loc15_43)
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
