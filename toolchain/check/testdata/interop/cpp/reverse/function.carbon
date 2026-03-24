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

// --- fail_todo_args.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_args.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:5:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with parameters` [SemanticsTodo]
// CHECK:STDERR: fn F3(_: i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
  // CHECK:STDERR: fail_todo_args.carbon:[[@LINE+4]]:18: error: no member named 'F3' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Carbon::Other::F3();
  // CHECK:STDERR:       |                  ^~
  // CHECK:STDERR:
  Carbon::Other::F3();
}
''';
