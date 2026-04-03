// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/reverse/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/reverse/function.carbon

// --- other.carbon
package Other;
fn NoArgs() {}
fn BoolArg(a: bool) {
  a;
}
fn IntArg(a: i32) {
  a;
}
fn FloatArg(a: f32) {
  a;
}
// --- function.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: function.carbon:[[@LINE+15]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:3:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with parameters` [SemanticsTodo]
// CHECK:STDERR: fn BoolArg(a: bool) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: function.carbon:[[@LINE+10]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:6:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with parameters` [SemanticsTodo]
// CHECK:STDERR: fn IntArg(a: i32) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: function.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:9:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with parameters` [SemanticsTodo]
// CHECK:STDERR: fn FloatArg(a: f32) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  Carbon::Other::NoArgs();
  // CHECK:STDERR: function.carbon:[[@LINE+4]]:18: error: no member named 'BoolArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:     8 |   Carbon::Other::BoolArg(true);
  // CHECK:STDERR:       |                  ^~~~~~~
  // CHECK:STDERR:
  Carbon::Other::BoolArg(true);
  // CHECK:STDERR: function.carbon:[[@LINE+4]]:18: error: no member named 'IntArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:     9 |   Carbon::Other::IntArg(123);
  // CHECK:STDERR:       |                  ^~~~~~
  // CHECK:STDERR:
  Carbon::Other::IntArg(123);
  // CHECK:STDERR: function.carbon:[[@LINE+4]]:18: error: no member named 'FloatArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    10 |   Carbon::Other::FloatArg(1.5);
  // CHECK:STDERR:       |                  ^~~~~~~~
  // CHECK:STDERR:
  Carbon::Other::FloatArg(1.5);
}
''';

