// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_generic_method.carbon

class I {}

class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

// TODO: The follow-on errors here aren't great. Investigate whether we can
// enter the scope anyway if the parameters don't match.
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+19]]:17: error: type `<pattern for I>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: fn Class(unused N:! I).F[self: Self](unused n: T) {}
// CHECK:STDERR:                 ^
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE-10]]:13: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+12]]:32: error: name `Self` not found [NameNotFound]
// CHECK:STDERR: fn Class(unused N:! I).F[self: Self](unused n: T) {}
// CHECK:STDERR:                                ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+8]]:48: error: name `T` not found [NameNotFound]
// CHECK:STDERR: fn Class(unused N:! I).F[self: Self](unused n: T) {}
// CHECK:STDERR:                                                ^
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+4]]:26: warning: binding `self` is unused [UnusedBinding]
// CHECK:STDERR: fn Class(unused N:! I).F[self: Self](unused n: T) {}
// CHECK:STDERR:                          ^~~~
// CHECK:STDERR:
fn Class(unused N:! I).F[self: Self](unused n: T) {}
