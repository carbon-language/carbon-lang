// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   12,
// CHECK:STDOUT:   24,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT:   d,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+8,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+0, arg1: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+2, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+6, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+9, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+10},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+12, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+12, arg1: node+14},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+16, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: ArrayIndex, arg0: node+2, arg1: node+18, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+16, arg1: node+20},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+22, type: type0},
// CHECK:STDOUT:   {kind: ArrayIndex, arg0: node+2, arg1: node+12, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+24, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+22, arg1: node+25},
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
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: [i32; 2] = var
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_24.1: i32 = int_literal 24
// CHECK:STDOUT:   %.loc7_24.2: i32 = stub_reference %.loc7_24.1
// CHECK:STDOUT:   %.loc7_26.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_26.2: (i32, i32) = tuple_value (%.loc7_20.2, %.loc7_24.2)
// CHECK:STDOUT:   %.loc7_26.3: [i32; 2] = array_value %.loc7_26.2
// CHECK:STDOUT:   assign %a, %.loc7_26.3
// CHECK:STDOUT:   %b: i32 = var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1
// CHECK:STDOUT:   assign %b, %.loc8
// CHECK:STDOUT:   %c: i32 = var
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_17.1: i32 = array_index %a, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = value_binding %.loc9_17.1
// CHECK:STDOUT:   assign %c, %.loc9_17.2
// CHECK:STDOUT:   %d: i32 = var
// CHECK:STDOUT:   %.loc10_17.1: i32 = array_index %a, %b
// CHECK:STDOUT:   %.loc10_17.2: i32 = value_binding %.loc10_17.1
// CHECK:STDOUT:   assign %d, %.loc10_17.2
// CHECK:STDOUT: }
