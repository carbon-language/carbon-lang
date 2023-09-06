// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   12,
// CHECK:STDOUT:   76,
// CHECK:STDOUT:   6,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+6,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+8,
// CHECK:STDOUT:   node+9,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type1},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock2, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock3, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str0, type: type4},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+11, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+13, type: type2},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+15, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type2},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type4},
// CHECK:STDOUT:   {kind: Assign, arg0: node+10, arg1: node+19},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_18.2: (type, type) = tuple_value (%.loc7_10, %.loc7_15)
// CHECK:STDOUT:   %.loc7_18.3: (type, type) = stub_reference %.loc7_18.2
// CHECK:STDOUT:   %.loc7_21: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_24.1: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %.loc7_24.2: ((type, type), type) = tuple_value (%.loc7_18.3, %.loc7_21)
// CHECK:STDOUT:   %.loc7_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_24.3: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %x: ref ((i32, i32), i32) = var "x"
// CHECK:STDOUT:   %.loc7_30.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_30.2: i32 = stub_reference %.loc7_30.1
// CHECK:STDOUT:   %.loc7_34.1: i32 = int_literal 76
// CHECK:STDOUT:   %.loc7_34.2: i32 = stub_reference %.loc7_34.1
// CHECK:STDOUT:   %.loc7_36.1: (i32, i32) = tuple_value (%.loc7_30.2, %.loc7_34.2)
// CHECK:STDOUT:   %.loc7_36.2: (i32, i32) = stub_reference %.loc7_36.1
// CHECK:STDOUT:   %.loc7_39.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_39.2: i32 = stub_reference %.loc7_39.1
// CHECK:STDOUT:   %.loc7_40: ((i32, i32), i32) = tuple_value (%.loc7_36.2, %.loc7_39.2)
// CHECK:STDOUT:   assign %x, %.loc7_40
// CHECK:STDOUT: }
