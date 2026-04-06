// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reverse/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reverse/function.carbon

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

// --- fail_todo_non_void.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_non_void.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:4:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with return type other than `()`` [SemanticsTodo]
// CHECK:STDERR: fn F2() -> i32 { return 0; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_non_void.carbon:[[@LINE+4]]:18: error: no member named 'F2' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Carbon::Other::F2();
  // CHECK:STDERR:       |                  ^~
  // CHECK:STDERR:
  Carbon::Other::F2();
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

import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_method.carbon:[[@LINE+8]]:3: error: invalid use of incomplete type 'Carbon::Other::C' [CppInteropParseError]
  // CHECK:STDERR:    15 |   Carbon::Other::C().Method();
  // CHECK:STDERR:       |   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_method.carbon:[[@LINE+4]]:21: error: member access into incomplete type 'Carbon::Other::C' [CppInteropParseError]
  // CHECK:STDERR:    15 |   Carbon::Other::C().Method();
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR:
  Carbon::Other::C().Method();
}
''';
