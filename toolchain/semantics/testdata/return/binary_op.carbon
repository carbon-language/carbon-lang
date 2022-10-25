// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   ident0 = "Main";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 12;
// CHECK:STDOUT:   int1 = 34;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = Identifier(ident0);
// CHECK:STDOUT:     node1 = FunctionDeclaration(node0);
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:     node0 = IntegerLiteral(int0);
// CHECK:STDOUT:     node1 = IntegerLiteral(int1);
// CHECK:STDOUT:     node2 = BinaryOperatorAdd(node0, node1);
// CHECK:STDOUT:     node3 = ReturnExpression(node2);
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Main() {
  return 12 + 34;
}
