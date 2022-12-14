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
// CHECK:STDOUT: },
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   ident0 = "x";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   int0 = 0;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = VarStorage(): node_xref2;
// CHECK:STDOUT:     node1 = BindName(ident0, node0): node_xref2;
// CHECK:STDOUT:     node2 = IntegerLiteral(int0): node_xref2;
// CHECK:STDOUT:     node3 = Assign(node0, node2): node_xref2;
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

var x: i32 = 0;
