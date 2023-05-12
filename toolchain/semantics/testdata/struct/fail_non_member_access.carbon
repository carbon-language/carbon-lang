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
// CHECK:STDOUT:   4,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
// CHECK:STDOUT:   b,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block1, arg1: block2, type: nodeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: node+1},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+2, type: node+1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StructType, arg0: block5, arg1: block5, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block4, arg1: block6, type: node+7},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+8, type: node+7},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+10, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+10, arg1: nodeInvalidType, type: nodeInvalidType},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var x: {.a: i32} = {.a = 4};
// CHECK:STDERR: toolchain/semantics/testdata/struct/fail_non_member_access.carbon:[[@LINE+1]]:15: Type `{.a: i32}` does not have a member `b`.
var y: i32 = x.b;
