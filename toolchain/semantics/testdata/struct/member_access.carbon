// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 0, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   y,
// CHECK:STDOUT:   z,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type1},
// CHECK:STDOUT:   {kind: StructType, arg0: block2, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+3, type: type2},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type1},
// CHECK:STDOUT:   {kind: StructValue, arg0: block4, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: node+11},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+13, type: type1},
// CHECK:STDOUT:   {kind: StructMemberAccess, arg0: node+3, arg1: member1, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+13, arg1: node+15},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+17, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+17, arg1: node+13},
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
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
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
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0: type = StructType {.a: f64, .b: i32}
// CHECK:STDOUT:   %x: {.a: f64, .b: i32} = VarStorage
// CHECK:STDOUT:   %1: f64 = RealLiteral 0e-1
// CHECK:STDOUT:   %2: f64 = StubReference %1
// CHECK:STDOUT:   %3: i32 = IntegerLiteral 1
// CHECK:STDOUT:   %4: i32 = StubReference %3
// CHECK:STDOUT:   %5: {.a: f64, .b: i32} = StructValue (%2, %4)
// CHECK:STDOUT:   Assign {.a: f64, .b: i32}, %x, %5
// CHECK:STDOUT:   %y: i32 = VarStorage
// CHECK:STDOUT:   %6: i32 = StructMemberAccess %x, member1
// CHECK:STDOUT:   Assign i32, %y, %6
// CHECK:STDOUT:   %z: i32 = VarStorage
// CHECK:STDOUT:   Assign i32, %z, %y
// CHECK:STDOUT: }

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;
