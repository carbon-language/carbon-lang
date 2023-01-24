// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Main,
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: block0, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: block1, type: node1},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: block2, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: block3, type: node0},
// CHECK:STDOUT:   {kind: FunctionDeclaration},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node4},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node4, arg1: block1},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node7, type: node2},
// CHECK:STDOUT:   {kind: RealLiteral, type: node3},
// CHECK:STDOUT:   {kind: Assign, arg0: node7, arg1: node9, type: node1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node4,
// CHECK:STDOUT:     node5,
// CHECK:STDOUT:     node6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node7,
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:     node9,
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Main() {
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/var/fail_init_type_mismatch.carbon:[[@LINE+1]]:19: Type mismatch: lhs is node2, rhs is node3
  var x: i32 = 1.0;
}
