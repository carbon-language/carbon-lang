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
fn IntReturn() -> i32 {
  return 123;
}
// --- function.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: function.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:12:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with return type other than `()`` [SemanticsTodo]
// CHECK:STDERR: fn IntReturn() -> i32 {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
int G() {
  Carbon::Other::NoArgs();
  Carbon::Other::BoolArg(true);
  Carbon::Other::IntArg(123);
  Carbon::Other::FloatArg(1.5);
  // CHECK:STDERR: function.carbon:[[@LINE+4]]:25: error: no member named 'IntReturn' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    20 |   return Carbon::Other::IntReturn();
  // CHECK:STDERR:       |                         ^~~~~~~~~
  // CHECK:STDERR:
  return Carbon::Other::IntReturn();
}
''';

// --- single_file.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {}

inline Cpp '''
inline void G1() {
  Carbon::F();
}

// Call an inline function indirectly to ensure declarations are getting
// properly registered with the CodeGen consumer.
inline void G2() {
  G1();
}
''';

fn H() { Cpp.G2(); }

