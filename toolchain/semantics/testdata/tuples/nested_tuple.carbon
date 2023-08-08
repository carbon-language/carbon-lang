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
// CHECK:STDOUT:   {kind: VarStorage, type: type4},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+10, type: type4},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+12, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+14, type: type2},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+16, type: type3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+18, type: type2},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type4},
// CHECK:STDOUT:   {kind: Assign, arg0: node+10, arg1: node+20},
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
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %7: type = stub_reference i32
// CHECK:STDOUT:   %7.15: type = stub_reference i32
// CHECK:STDOUT:   %7.18: type = tuple_type (type, type)
// CHECK:STDOUT:   %7.18.0: (type, type) = tuple_value (%7, %7.15)
// CHECK:STDOUT:   %7.18.1: (type, type) = stub_reference %7.18.0
// CHECK:STDOUT:   %7.21: type = stub_reference i32
// CHECK:STDOUT:   %7.24: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %7.24.0: ((type, type), type) = tuple_value (%7.18.1, %7.21)
// CHECK:STDOUT:   %7.18.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %7.24.1: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %x: ((i32, i32), i32) = var
// CHECK:STDOUT:   %7.30: i32 = int_value 12
// CHECK:STDOUT:   %7.30.0: i32 = stub_reference %7.30
// CHECK:STDOUT:   %7.34: i32 = int_value 76
// CHECK:STDOUT:   %7.34.0: i32 = stub_reference %7.34
// CHECK:STDOUT:   %7.36: (i32, i32) = tuple_value (%7.30.0, %7.34.0)
// CHECK:STDOUT:   %7.36.0: (i32, i32) = stub_reference %7.36
// CHECK:STDOUT:   %7.39: i32 = int_value 6
// CHECK:STDOUT:   %7.39.0: i32 = stub_reference %7.39
// CHECK:STDOUT:   %7.40: ((i32, i32), i32) = tuple_value (%7.36.0, %7.39.0)
// CHECK:STDOUT:   assign %x, %7.40
// CHECK:STDOUT: }
