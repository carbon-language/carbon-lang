// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   %ident{0} = "Main";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT:   %int{0} = 12;
// CHECK:STDOUT:   %int{1} = 34;
// CHECK:STDOUT: },
// CHECK:STDOUT: node_blocks = {
// CHECK:STDOUT:   %block{0} = {
// CHECK:STDOUT:     %0 = Identifier(%ident{0});
// CHECK:STDOUT:     %1 = FunctionDeclaration(%0);
// CHECK:STDOUT:   },
// CHECK:STDOUT:   %block{1} = {
// CHECK:STDOUT:     %0 = IntegerLiteral(%int{0});
// CHECK:STDOUT:     %1 = IntegerLiteral(%int{1});
// CHECK:STDOUT:     %2 = BinaryOperatorAdd(%0, %1);
// CHECK:STDOUT:     %3 = ReturnExpression(%2);
// CHECK:STDOUT:   },
// CHECK:STDOUT: }

fn Main() {
  return 12 + 34;
}
