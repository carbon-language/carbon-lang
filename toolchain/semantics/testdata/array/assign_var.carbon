// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   3,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   node+16,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str0, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+7, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+9, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+11, type: type1},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+6, arg1: node+13},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type1},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+15, arg1: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type3},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+6, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+18, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+17, arg1: node+19},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_19: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc7_22.2: (type, type, type) = tuple_value (%.loc7_9, %.loc7_14, %.loc7_19)
// CHECK:STDOUT:   %.loc7_22.3: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %a: (i32, i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_27.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_27.2: i32 = stub_reference %.loc7_27.1
// CHECK:STDOUT:   %.loc7_30.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_30.2: i32 = stub_reference %.loc7_30.1
// CHECK:STDOUT:   %.loc7_33.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_33.2: i32 = stub_reference %.loc7_33.1
// CHECK:STDOUT:   %.loc7_34: (i32, i32, i32) = tuple_value (%.loc7_27.2, %.loc7_30.2, %.loc7_33.2)
// CHECK:STDOUT:   assign %a, %.loc7_34
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, i32
// CHECK:STDOUT:   %b: [i32; 3] = var "b"
// CHECK:STDOUT:   %.loc7_5.1: (i32, i32, i32) = value_binding %a
// CHECK:STDOUT:   %.loc7_5.2: [i32; 3] = array_value %.loc7_5.1
// CHECK:STDOUT:   assign %b, %.loc7_5.2
// CHECK:STDOUT: }
