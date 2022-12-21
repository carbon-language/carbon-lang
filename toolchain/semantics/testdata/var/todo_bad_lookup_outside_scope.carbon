// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs.size == 2,
// CHECK:STDOUT: cross_references = {
// CHECK:STDOUT:   node_xref0 = xref(ir0, block0, node0);
// CHECK:STDOUT:   node_xref1 = xref(ir0, block0, node1);
// CHECK:STDOUT:   node_xref2 = xref(ir0, block0, node2);
// CHECK:STDOUT:   node_xref3 = xref(ir0, block0, node3);
// CHECK:STDOUT:   node_xref4 = xref(ir1, block0, node1);
// CHECK:STDOUT:   node_xref5 = xref(ir1, block1, node1);
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT: },
// CHECK:STDOUT: strings = {
// CHECK:STDOUT:   str0 = "Main";
// CHECK:STDOUT:   str1 = "x";
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = FunctionDeclaration();
// CHECK:STDOUT:     node1 = BindName(str0, node0);
// CHECK:STDOUT:     node2 = FunctionDefinition(node0, block1);
// CHECK:STDOUT:   },
// CHECK:STDOUT:   block1 = {
// CHECK:STDOUT:     node0 = VarStorage(): node_xref2;
// CHECK:STDOUT:     node1 = BindName(str1, node0): node_xref2;
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Main() {
  var x: i32;
}

// TODO: This is a crash because the type of `x` is incorrect. That needs to be
// fixed, but it crashes due to cross-reference handling which I intend to
// rewrite, so leaving this as a TODO.
// var y: i32 = x;
