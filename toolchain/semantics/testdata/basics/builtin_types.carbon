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
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+2, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+0, arg1: node+3},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+5, type: type1},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+7, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+5, arg1: node+8},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+10, type: type2},
// CHECK:STDOUT:   {kind: StringLiteral, arg0: str3, type: type2},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+12, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+10, arg1: node+13},
// CHECK:STDOUT:   {kind: VarStorage, type: typeTypeType},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+15, type: typeTypeType},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+15, arg1: node+17},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %test_i32: i32 = var
// CHECK:STDOUT:   %.loc7_21.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_21.2: i32 = initialize_from %.loc7_21.1
// CHECK:STDOUT:   assign %test_i32, %.loc7_21.2
// CHECK:STDOUT:   %test_f64: f64 = var
// CHECK:STDOUT:   %.loc8_21.1: f64 = real_literal 1e-1
// CHECK:STDOUT:   %.loc8_21.2: f64 = initialize_from %.loc8_21.1
// CHECK:STDOUT:   assign %test_f64, %.loc8_21.2
// CHECK:STDOUT:   %test_str: String = var
// CHECK:STDOUT:   %.loc9_24.1: String = string_literal "Test"
// CHECK:STDOUT:   %.loc9_24.2: String = initialize_from %.loc9_24.1
// CHECK:STDOUT:   assign %test_str, %.loc9_24.2
// CHECK:STDOUT:   %test_type: type = var
// CHECK:STDOUT:   %.1: type = initialize_from i32
// CHECK:STDOUT:   assign %test_type, %.1
// CHECK:STDOUT: }
