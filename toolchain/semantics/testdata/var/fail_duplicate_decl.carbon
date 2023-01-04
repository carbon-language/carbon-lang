// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs.size == 1,
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 0;
// CHECK:STDOUT:   int1 = 0;
// CHECK:STDOUT: },
// CHECK:STDOUT: strings = {
// CHECK:STDOUT:   str0 = "Main";
// CHECK:STDOUT:   str1 = "x";
// CHECK:STDOUT: },
// CHECK:STDOUT: nodes = {
// CHECK:STDOUT:   node0 = CrossReference(ir0, block0): node0;
// CHECK:STDOUT:   node1 = CrossReference(ir0, block1): node1;
// CHECK:STDOUT:   node2 = CrossReference(ir0, block2): node0;
// CHECK:STDOUT:   node3 = CrossReference(ir0, block3): node0;
// CHECK:STDOUT:   node4 = FunctionDeclaration();
// CHECK:STDOUT:   node5 = BindName(str0, node4);
// CHECK:STDOUT:   node6 = FunctionDefinition(node4, block1);
// CHECK:STDOUT:   node7 = VarStorage(): node2;
// CHECK:STDOUT:   node8 = BindName(str1, node7): node2;
// CHECK:STDOUT:   node9 = IntegerLiteral(int0): node2;
// CHECK:STDOUT:   node10 = Assign(node7, node9): node2;
// CHECK:STDOUT:   node11 = VarStorage(): node2;
// CHECK:STDOUT:   node12 = BindName(str1, node11): node2;
// CHECK:STDOUT:   node13 = IntegerLiteral(int1): node2;
// CHECK:STDOUT:   node14 = Assign(node7, node13): node2;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node4;
// CHECK:STDOUT:     node5;
// CHECK:STDOUT:     node6;
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:     node7;
// CHECK:STDOUT:     node8;
// CHECK:STDOUT:     node9;
// CHECK:STDOUT:     node10;
// CHECK:STDOUT:     node11;
// CHECK:STDOUT:     node12;
// CHECK:STDOUT:     node13;
// CHECK:STDOUT:     node14;
// CHECK:STDOUT:   },
// CHECK:STDOUT: }


fn Main() {
  var x: i32 = 0;
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/var/fail_duplicate_decl.carbon:[[@LINE+2]]:7: Redefining x in the same scope.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/var/fail_duplicate_decl.carbon:[[@LINE-2]]:7: Previous definition is here.
  var x: i32 = 0;
}
