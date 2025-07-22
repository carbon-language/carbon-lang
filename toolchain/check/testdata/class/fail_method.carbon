// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_method.carbon

class Class {
  fn NoSelf();
  fn WithSelf[self: Class]();
}

alias A = Class.WithSelf;

fn F(c: Class) {
  c.NoSelf();
  c.WithSelf();

  Class.NoSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-13]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-21]]:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf(c);

  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-30]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A();
}
