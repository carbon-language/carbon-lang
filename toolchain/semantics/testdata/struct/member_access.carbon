// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 0, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
// CHECK:STDOUT:   z,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block1, arg1: block2, type: nodeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: node+2},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+3, type: node+2},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block5, arg1: block5, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block4, arg1: block6, type: node+11},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: node+12, type: node+11},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+14, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+3, arg1: member1, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+14, arg1: node+16, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+18, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+18, arg1: node+14, type: nodeIntegerType},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
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
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;
