// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   ident0 = "Foo";
// CHECK:STDOUT:   ident1 = "Bar";
// CHECK:STDOUT:   ident2 = "Baz";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   block0 = {
// CHECK:STDOUT:     node0 = Identifier(ident0);
// CHECK:STDOUT:     node1 = FunctionDeclaration(node0);
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
