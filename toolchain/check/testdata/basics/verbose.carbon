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
// CHECK:STDERR: AddNode: {kind: FunctionDeclaration, arg0: {{.*}}, type: type{{[0-9]+}}}
// CHECK:STDERR: node_block_stack_ Push 1
// CHECK:STDERR: AddNode: {kind: Return}
// CHECK:STDERR: node_block_stack_ Pop 1: block{{[0-9]+}}

fn Foo() {
  return;
}
