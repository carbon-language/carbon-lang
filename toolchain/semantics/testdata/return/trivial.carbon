// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: identifiers = {
// CHECK:STDOUT:   %s0 = "Main";
// CHECK:STDOUT: },
// CHECK:STDOUT: integer_literals = {
// CHECK:STDOUT: },
// CHECK:STDOUT: nodes = {
// CHECK:STDOUT:   %0 = Identifier(%s0);
// CHECK:STDOUT:   %1 = FunctionDeclaration(%0);
// CHECK:STDOUT:   %2 = FunctionDefinitionStart(%1);
// CHECK:STDOUT:     %3 = Return();
// CHECK:STDOUT:   %4 = FunctionDefinitionEnd();
// CHECK:STDOUT: }

fn Main() {
  return;
}
