// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOAUTOUPDATE
// RUN: %{carbon} -v dump semantics-ir %s | %{FileCheck-allow-unmatched}
//
// Only checks a couple statements in order to minimize manual update churn.
// CHECK:STDERR: Push 0: FunctionIntroducer -> node<invalid>
// CHECK:STDERR: AddNode block0: {kind: BindName, arg0: str0, arg1: node4}

fn Foo() {
  return;
}
