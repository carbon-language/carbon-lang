// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
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
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node0, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node1, type: node1},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node2, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node3, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node4, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node5, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node6, type: node5},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node7, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: node2},
// CHECK:STDOUT:   {kind: Assign, arg0: node7, arg1: node9, type: node2},
// CHECK:STDOUT:   {kind: VarStorage, type: node3},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node11, type: node3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: node3},
// CHECK:STDOUT:   {kind: Assign, arg0: node11, arg1: node13, type: node3},
// CHECK:STDOUT:   {kind: VarStorage, type: node4},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node15, type: node4},
// CHECK:STDOUT:   {kind: StringLiteral, arg0: str3, type: node4},
// CHECK:STDOUT:   {kind: Assign, arg0: node15, arg1: node17, type: node4},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node7,
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:     node9,
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:     node11,
// CHECK:STDOUT:     node12,
// CHECK:STDOUT:     node13,
// CHECK:STDOUT:     node14,
// CHECK:STDOUT:     node15,
// CHECK:STDOUT:     node16,
// CHECK:STDOUT:     node17,
// CHECK:STDOUT:     node18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";
