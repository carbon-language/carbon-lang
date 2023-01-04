// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs.size == 1,
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 12;
// CHECK:STDOUT: },
// CHECK:STDOUT: strings = {
// CHECK:STDOUT:   str0 = "Main";
// CHECK:STDOUT: },
// CHECK:STDOUT: nodes = {
// CHECK:STDOUT:   node0 = CrossReference(ir0, block0): node0;
// CHECK:STDOUT:   node1 = CrossReference(ir0, block1): node1;
// CHECK:STDOUT:   node2 = CrossReference(ir0, block2): node0;
// CHECK:STDOUT:   node3 = CrossReference(ir0, block3): node0;
// CHECK:STDOUT:   node4 = FunctionDeclaration();
// CHECK:STDOUT:   node5 = BindName(str0, node4);
// CHECK:STDOUT:   node6 = FunctionDefinition(node4, block1);
// CHECK:STDOUT:   node7 = IntegerLiteral(int0): node2;
// CHECK:STDOUT:   node8 = RealLiteral(): node3;
// CHECK:STDOUT:   node9 = BinaryOperatorAdd(node7, node8): node1;
// CHECK:STDOUT:   node10 = ReturnExpression(node9): node1;
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
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Main() {
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/operators/fail_type_mismatch.carbon:[[@LINE+1]]:13: Type mismatch: lhs is node2, rhs is node3
  return 12 + 3.4;
}
