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
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+0, arg1: node+2, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+4, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+4, arg1: node+6, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeStringType},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+8, type: nodeStringType},
// CHECK:STDOUT:   {kind: StringLiteral, arg0: str3, type: nodeStringType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+8, arg1: node+10, type: nodeStringType},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";
