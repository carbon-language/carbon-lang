// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_todo_form_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_todo_form_thunk.carbon

interface I {
  let T:! type;
  // CHECK:STDERR: fail_todo_form_thunk.carbon:[[@LINE+7]]:8: error: semantics TODO: `Support for cloning form bindings` [SemanticsTodo]
  // CHECK:STDERR:   fn F(x:? form(ref ()), y: T);
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_form_thunk.carbon:[[@LINE+3]]:8: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn F(x:? form(ref ()), y: T);
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~
  fn F(x:? form(ref ()), y: T);
}

class C {
  impl as I where .T = () {
    // CHECK:STDERR: fail_todo_form_thunk.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(x:? form(ref ()), ());
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_todo_form_thunk.carbon:[[@LINE-8]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(x:? form(ref ()), y: T);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(x:? form(ref ()), ());
  }
}
