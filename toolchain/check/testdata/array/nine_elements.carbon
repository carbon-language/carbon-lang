// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   -7,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   6,
// CHECK:STDOUT:   7,
// CHECK:STDOUT:   -8,
// CHECK:STDOUT:   -7,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+21,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+0, arg1: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str0, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+3, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+7, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+9, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+11, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int6, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+13, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int7, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+15, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int8, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int9, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+22, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+23},
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
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal -7
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 9] = var "a"
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_23.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_23.2: i32 = stub_reference %.loc7_23.1
// CHECK:STDOUT:   %.loc7_26.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_26.2: i32 = stub_reference %.loc7_26.1
// CHECK:STDOUT:   %.loc7_29.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_29.2: i32 = stub_reference %.loc7_29.1
// CHECK:STDOUT:   %.loc7_32.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc7_32.2: i32 = stub_reference %.loc7_32.1
// CHECK:STDOUT:   %.loc7_35.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_35.2: i32 = stub_reference %.loc7_35.1
// CHECK:STDOUT:   %.loc7_38.1: i32 = int_literal 7
// CHECK:STDOUT:   %.loc7_38.2: i32 = stub_reference %.loc7_38.1
// CHECK:STDOUT:   %.loc7_41.1: i32 = int_literal -8
// CHECK:STDOUT:   %.loc7_41.2: i32 = stub_reference %.loc7_41.1
// CHECK:STDOUT:   %.loc7_44.1: i32 = int_literal -7
// CHECK:STDOUT:   %.loc7_44.2: i32 = stub_reference %.loc7_44.1
// CHECK:STDOUT:   %.loc7_45.1: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc7_45.2: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = tuple_value (%.loc7_20.2, %.loc7_23.2, %.loc7_26.2, %.loc7_29.2, %.loc7_32.2, %.loc7_35.2, %.loc7_38.2, %.loc7_41.2, %.loc7_44.2)
// CHECK:STDOUT:   %.loc7_45.3: [i32; 9] = array_value %.loc7_45.2
// CHECK:STDOUT:   assign %a, %.loc7_45.3
// CHECK:STDOUT: }
