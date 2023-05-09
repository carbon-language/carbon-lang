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
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block1, arg1: block2, type: nodeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: node+2},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+3, type: node+2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block5, arg1: block5, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block4, arg1: block6, type: node+8},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: nodeInvalidType, type: nodeInvalidType},
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
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/semantics/testdata/struct/fail_too_few_values.carbon:[[@LINE+1]]:37: Cannot implicitly convert from `{.a: i32}` to `{.a: i32, .b: i32}`.
var x: {.a: i32, .b: i32} = {.a = 1};
