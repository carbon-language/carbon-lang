// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT:   {arg_ir: block0, arg_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Foo,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node0, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node1, type: node1},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node2, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node3, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node4, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node5, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node6, type: node5},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str0, arg1: callable0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node7, arg1: block0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable1},
// CHECK:STDOUT:   {kind: Call, arg0: call0, arg1: callable0, type: node6},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node9, arg1: block2},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node7,
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:     node9,
// CHECK:STDOUT:     node11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Foo() {}

fn Main() {
  Foo();
}
