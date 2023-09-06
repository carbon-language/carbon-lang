// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: -v compile --phase=check %s
//
// Only checks a couple statements in order to minimize manual update churn.
// NOAUTOUPDATE
// SET-CHECK-SUBSET
// CHECK:STDERR: Node Push 0: FunctionIntroducer -> <none>
// CHECK:STDERR: AddNode block{{[0-9]+}}: {kind: FunctionDeclaration, arg0: function{{[0-9]+}}}

fn Foo() {
  return;
}
