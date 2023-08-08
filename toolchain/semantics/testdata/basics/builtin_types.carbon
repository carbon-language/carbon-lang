// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";
var test_type: type = i32;

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 1, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   test_i32,
// CHECK:STDOUT:   test_f64,
// CHECK:STDOUT:   test_str,
// CHECK:STDOUT:   Test,
// CHECK:STDOUT:   test_type,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT:   nodeStringType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+0, arg1: node+2},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+4, type: type1},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+4, arg1: node+6},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+8, type: type2},
// CHECK:STDOUT:   {kind: StringLiteral, arg0: str3, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+8, arg1: node+10},
// CHECK:STDOUT:   {kind: VarStorage, type: typeTypeType},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+12, type: typeTypeType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+12, arg1: nodeIntegerType},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %test_i32: i32 = var
// CHECK:STDOUT:   %.L7: i32 = int_value 0
// CHECK:STDOUT:   assign %test_i32, %.L7
// CHECK:STDOUT:   %test_f64: f64 = var
// CHECK:STDOUT:   %.L8: f64 = real_value 1e-1
// CHECK:STDOUT:   assign %test_f64, %.L8
// CHECK:STDOUT:   %test_str: String = var
// CHECK:STDOUT:   %.L9: String = string_value "Test"
// CHECK:STDOUT:   assign %test_str, %.L9
// CHECK:STDOUT:   %test_type: type = var
// CHECK:STDOUT:   assign %test_type, i32
// CHECK:STDOUT: }
