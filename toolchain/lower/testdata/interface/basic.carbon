// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/basic.carbon

interface I {
  fn Assoc();
}

// There are no interesting runtime operations here, but there's no rule saying
// you can't pass a facet around at runtime, so make sure it works.
fn F(T: I) -> I { return T; }

interface J;

// Declared-but-not-defined interfaces are still complete types.
// CHECK:STDERR: basic.carbon:[[@LINE+7]]:6: error: parameter has incomplete type `J` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn G(T: J) {}
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: basic.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn G(T: J) {}
