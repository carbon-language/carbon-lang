// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs.size == 1,
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 0;
// CHECK:STDOUT: },
// CHECK:STDOUT: strings = {
// CHECK:STDOUT:   str0 = "x";
// CHECK:STDOUT:   str1 = "Main";
// CHECK:STDOUT:   str2 = "y";
// CHECK:STDOUT: },
// CHECK:STDOUT: nodes = {
// CHECK:STDOUT:   node0 = CrossReference(ir0, block0): node0;
// CHECK:STDOUT:   node1 = CrossReference(ir0, block1): node1;
// CHECK:STDOUT:   node2 = CrossReference(ir0, block2): node0;
// CHECK:STDOUT:   node3 = CrossReference(ir0, block3): node0;
// CHECK:STDOUT:   node4 = VarStorage(): node2;
// CHECK:STDOUT:   node5 = BindName(str0, node4): node2;
// CHECK:STDOUT:   node6 = IntegerLiteral(int0): node2;
// CHECK:STDOUT:   node7 = Assign(node4, node6): node2;
// CHECK:STDOUT:   node8 = FunctionDeclaration();
// CHECK:STDOUT:   node9 = BindName(str1, node8);
// CHECK:STDOUT:   node10 = FunctionDefinition(node8, block1);
// CHECK:STDOUT:   node11 = VarStorage(): node2;
// CHECK:STDOUT:   node12 = BindName(str2, node11): node2;
// CHECK:STDOUT:   node13 = Assign(node11, node4): node2;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node4;
// CHECK:STDOUT:     node5;
// CHECK:STDOUT:     node6;
// CHECK:STDOUT:     node7;
// CHECK:STDOUT:     node8;
// CHECK:STDOUT:     node9;
// CHECK:STDOUT:     node10;
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:     node11;
// CHECK:STDOUT:     node12;
// CHECK:STDOUT:     node13;
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

var x: i32 = 0;

fn Main() {
  var y: i32 = x;
}
