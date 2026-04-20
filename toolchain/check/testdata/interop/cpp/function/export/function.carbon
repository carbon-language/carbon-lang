// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/function.carbon

// --- other.carbon

package Other;
fn F1() {}
fn F2() -> i32 { return 0; }
fn F3(_: i32) {}

fn HasGenericArg(T:! type, a: T) { a; }
fn HasDeducedArg[T:! type](a: T) { a; }

class C {
  fn Method[self: Self]() { self; }
}

// --- function.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void G() {
  Carbon::Other::F1();
}
''';

// --- return_int.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int G() {
  return Carbon::Other::F2();
}
''';

// --- args.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void G() {
  Carbon::Other::F3(123);
}
''';

// --- fail_todo_generic.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:7:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with generic parameters` [SemanticsTodo]
// CHECK:STDERR: fn HasGenericArg(T:! type, a: T) { a; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+8]]:18: error: no member named 'HasGenericArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::Other::HasGenericArg<int>(123);
  // CHECK:STDERR:       |                  ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+4]]:35: error: expected '(' for function-style cast or type construction [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::Other::HasGenericArg<int>(123);
  // CHECK:STDERR:       |                                ~~~^
  // CHECK:STDERR:
  Carbon::Other::HasGenericArg<int>(123);
}
''';

// --- fail_todo_deduced.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_deduced.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:8:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with generic parameters` [SemanticsTodo]
// CHECK:STDERR: fn HasDeducedArg[T:! type](a: T) { a; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_deduced.carbon:[[@LINE+4]]:18: error: no member named 'HasDeducedArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Carbon::Other::HasDeducedArg(123);
  // CHECK:STDERR:       |                  ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Carbon::Other::HasDeducedArg(123);
}
''';

// --- fail_todo_method.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_method.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:11:3: error: semantics TODO: `unsupported: C++ calling a Carbon function with an implicit parameter` [SemanticsTodo]
// CHECK:STDERR:   fn Method[self: Self]() { self; }
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_method.carbon:[[@LINE+4]]:22: error: no member named 'Method' in 'Carbon::Other::C' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Carbon::Other::C().Method();
  // CHECK:STDERR:       |   ~~~~~~~~~~~~~~~~~~ ^
  // CHECK:STDERR:
  Carbon::Other::C().Method();
}
''';
