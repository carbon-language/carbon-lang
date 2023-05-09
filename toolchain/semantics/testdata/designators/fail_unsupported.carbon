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
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
// CHECK:STDOUT:   b,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+2, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: nodeInvalidType, type: nodeInvalidType},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var x: i32;
// CHECK:STDERR: toolchain/semantics/testdata/designators/fail_unsupported.carbon:[[@LINE+1]]:15: Type `i32` does not support designator expressions.
var y: i32 = x.b;
