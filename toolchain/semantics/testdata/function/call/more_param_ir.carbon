// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT:   {arg_ir: block4, arg_refs: block5},
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_ir: block1, param_refs: block2},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   6,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT:   Foo,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+2, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+4, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str3, arg1: callable0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+6, arg1: block0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str4, arg1: callable1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+9, arg1: node+10, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+11, arg1: node+12, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+13, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+15, arg1: node+16, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Call, arg0: call0, arg1: callable0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+8, arg1: block6},
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
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Foo(a: i32, b: i32, c: i32) {}

fn Main() {
  // Generates multiple IR instructions for the first two parameters.
  Foo(1 + 2 + 3, 4 + 5, 6);
}
