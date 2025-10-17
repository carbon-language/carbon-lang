// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/fail_todo_unspecified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/fail_todo_unspecified.carbon

interface Z {}

constraint E {
  // CHECK:STDERR: fail_todo_unspecified.carbon:[[@LINE+8]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_unspecified.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Z;
}

fn F(T:! E) {}

class C {}
impl C as Z {}

fn G(T:! Z) {
  // Anything that impls Z also impls E, since that's all it requires. Nothing
  // needs to specify that T impls E.
  F(T);
  // Same for concrete types.
  F(C);
}
