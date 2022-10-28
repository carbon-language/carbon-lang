// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs.size == 2,
// CHECK:STDOUT: cross_references = {
// CHECK:STDOUT:   node_xref0 = "xref(ir0, block0, node0)";
// CHECK:STDOUT:   node_xref1 = "xref(ir0, block0, node1)";
// CHECK:STDOUT: },
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   ident0 = "Foo";
// CHECK:STDOUT:   ident1 = "Bar";
// CHECK:STDOUT:   ident2 = "Baz";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = FunctionDeclaration();
// CHECK:STDOUT:     node1 = BindName(ident0, node0);
// CHECK:STDOUT:     node2 = FunctionDefinition(node0, block1);
// CHECK:STDOUT:     node3 = FunctionDeclaration();
// CHECK:STDOUT:     node4 = BindName(ident1, node3);
// CHECK:STDOUT:     node5 = FunctionDefinition(node3, block2);
// CHECK:STDOUT:     node6 = FunctionDeclaration();
// CHECK:STDOUT:     node7 = BindName(ident2, node6);
// CHECK:STDOUT:     node8 = FunctionDefinition(node6, block3);
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block2 = {
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block3 = {
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Foo() {}
fn Bar() {}
fn Baz() {}
