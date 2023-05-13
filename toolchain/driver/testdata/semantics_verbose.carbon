// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} -v dump semantics-ir %s | %{FileCheck-allow-unmatched}
//
// Only checks a couple statements in order to minimize manual update churn.
// CHECK:STDERR: Node Push 0: FunctionIntroducer -> <none>
// CHECK:STDERR: AddNode block{{[0-9]+}}: {kind: FunctionDeclaration, arg0: str{{[0-9]+}}, arg1: callable{{[0-9]+}}}

fn Foo() {
  return;
}
