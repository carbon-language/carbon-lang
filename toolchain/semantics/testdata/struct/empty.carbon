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
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+0, arg1: nodeEmptyStruct, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+3, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: node+0, type: nodeEmptyStructType},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

var x: {} = {};
var y: {} = x;
