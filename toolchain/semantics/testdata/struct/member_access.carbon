// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

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
// CHECK:STDOUT:   {kind: VarStorage, arg0: str0, type: type2},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+7, type: type1},
// CHECK:STDOUT:   {kind: StructValue, arg0: block4, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: node+10},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str3, type: type1},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+3, arg1: member1, type: type1},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+13, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+12, arg1: node+14},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str4, type: type1},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+12, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+16, arg1: node+17},
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
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_25: type = struct_type {.a: f64, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: f64, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc7_35: f64 = real_literal 0e-1
// CHECK:STDOUT:   %.loc7_33: f64 = stub_reference %.loc7_35
// CHECK:STDOUT:   %.loc7_45: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_43: i32 = stub_reference %.loc7_45
// CHECK:STDOUT:   %.loc7_46: {.a: f64, .b: i32} = struct_value (%.loc7_33, %.loc7_43)
// CHECK:STDOUT:   assign %x, %.loc7_46
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %.loc8_15.1: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc8_15.2: i32 = bind_value %.loc8_15.1
// CHECK:STDOUT:   assign %y, %.loc8_15.2
// CHECK:STDOUT:   %z: ref i32 = var "z"
// CHECK:STDOUT:   %.loc8_5: i32 = bind_value %y
// CHECK:STDOUT:   assign %z, %.loc8_5
// CHECK:STDOUT: }
