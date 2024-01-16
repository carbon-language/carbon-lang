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
// CHECK:STDERR: AddPlaceholderInst: {kind: FunctionDecl, arg0: {{.*}}, type: type{{[0-9]+}}}
// CHECK:STDERR: ReplaceInst: inst+{{[0-9]+}} -> {kind: FunctionDecl, arg0: {{.*}}, type: type{{[0-9]+}}}
// CHECK:STDERR: inst_block_stack_ Push 1
// CHECK:STDERR: AddInst: {kind: Return}
// CHECK:STDERR: inst_block_stack_ Pop 1: block{{[0-9]+}}

fn Foo() {
  return;
}
