// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b[0];

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   12,
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+4, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+6, type: type1},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type2},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+8, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+4, arg1: node+9},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+13, type: type2},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+4, type: type2},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+15, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+13, arg1: node+16},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+18, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+13, arg1: node+20, type: type1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+21, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+22, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+18, arg1: node+23},
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
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_13.2: (type,) = tuple_value (%.loc7_9)
// CHECK:STDOUT:   %.loc7_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %a: (i32,) = var
// CHECK:STDOUT:   %.loc7_18.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_18.2: i32 = stub_reference %.loc7_18.1
// CHECK:STDOUT:   %.loc7_21.1: (i32,) = tuple_value (%.loc7_18.2)
// CHECK:STDOUT:   %.loc7_21.2: (i32,) = initialize_from %.loc7_21.1
// CHECK:STDOUT:   assign %a, %.loc7_21.2
// CHECK:STDOUT:   %.loc8_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_13: (type,) = tuple_value (%.loc8_9)
// CHECK:STDOUT:   %b: (i32,) = var
// CHECK:STDOUT:   %.loc7_5.1: (i32,) = value_binding %a
// CHECK:STDOUT:   %.loc7_5.2: (i32,) = initialize_from %.loc7_5.1
// CHECK:STDOUT:   assign %b, %.loc7_5.2
// CHECK:STDOUT:   %c: i32 = var
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_17.1: i32 = tuple_index %b, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = value_binding %.loc9_17.1
// CHECK:STDOUT:   %.loc9_17.3: i32 = initialize_from %.loc9_17.2
// CHECK:STDOUT:   assign %c, %.loc9_17.3
// CHECK:STDOUT: }
