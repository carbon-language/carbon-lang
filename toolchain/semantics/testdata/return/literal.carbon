// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   ident0 = "Type";
// CHECK:STDOUT:   ident1 = "Main";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 0;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = Builtin(TypeType): node0;
// CHECK:STDOUT:     node1 = BindName(ident0, node0);
// CHECK:STDOUT:     node2 = Builtin(Int32): node0;
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:     node0 = FunctionDeclaration();
// CHECK:STDOUT:     node1 = BindName(ident1, node0);
// CHECK:STDOUT:     node2 = FunctionDefinition(node0, block2);
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block2 = {
// CHECK:STDOUT:     node0 = IntegerLiteral(int0);
// CHECK:STDOUT:     node1 = ReturnExpression(node0);
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Main() {
  return 0;
}
