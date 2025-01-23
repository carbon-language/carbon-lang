// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: -v compile --no-prelude-import --phase=check %s
//
// Only checks a couple statements in order to minimize manual update churn.
// To test this file alone, run:
//   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/verbose.carbon
// To dump output, run:
//   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/verbose.carbon
// NOAUTOUPDATE
// SET-CHECK-SUBSET
// CHECK:STDERR: Node Push 0: FunctionIntroducer -> <none>
// CHECK:STDERR: AddPlaceholderInst: {kind: FunctionDecl, arg0: function<none>, arg1: inst_block_empty}
// CHECK:STDERR: ReplaceInst: inst{{[0-9]+}} -> {kind: FunctionDecl, arg0: function{{[0-9]+}}, arg1: inst_block_empty, type: type(inst{{[0-9]+}})}
// CHECK:STDERR: inst_block_stack_ Push 1
// CHECK:STDERR: AddInst: {kind: Return}
// CHECK:STDERR: inst_block_stack_ Pop 1: inst_block{{[0-9]+}}

fn Foo() {
  return;
}
