// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/fail_invalid_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/fail_invalid_method.carbon

constraint E {
  // TODO: Any fn in a non-template constraint is an error.
  fn DeclaredMethod();
  fn DeclaredMethodWithSelf[self: Self](b: Self) -> Self;
}

constraint Foo {
  // TODO: Any fn in a non-template constraint is an error.
  fn DefinedMethod() {}
  // CHECK:STDERR: fail_invalid_method.carbon:[[@LINE+4]]:58: error: missing `return` at end of function with declared return type [MissingReturnStatement]
  // CHECK:STDERR:   fn DefinedMethodWithSelf[self: Self](b: Self) -> Self {}
  // CHECK:STDERR:                                                          ^
  // CHECK:STDERR:
  fn DefinedMethodWithSelf[self: Self](b: Self) -> Self {}
}
