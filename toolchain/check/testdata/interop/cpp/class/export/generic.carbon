// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/generic.carbon

// --- generic_class.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C(T: type) {
  var t: T;
}

inline Cpp '''
void F() {
  // CHECK:STDERR: generic_class.carbon:[[@LINE+7]]:18: error: implicit instantiation of undefined template 'Carbon::C<int>' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Carbon::C<int> c;
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR: generic_class.carbon:[[@LINE-9]]:18: note: template is declared here [CppInteropParseNote]
  // CHECK:STDERR:     4 | class C(T: type) {
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  Carbon::C<int> c;
  c.t = 123;
}
''';

// --- fail_todo_specific_alias.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_todo_specific_alias.carbon:[[@LINE+4]]:1: error: semantics TODO: `interop with specific class` [SemanticsTodo]
// CHECK:STDERR: class C(T: type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T: type) {
  var t: T;
}
alias A = C(i32);

inline Cpp '''
void F() {
  // CHECK:STDERR: fail_todo_specific_alias.carbon:[[@LINE+8]]:11: error: semantics TODO: `interop with unsupported type` [SemanticsTodo]
  // CHECK:STDERR:   Carbon::A a;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_specific_alias.carbon:[[@LINE+4]]:11: error: no type named 'A' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    23 |   Carbon::A a;
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::A a;
  // CHECK:STDERR: fail_todo_specific_alias.carbon:[[@LINE+4]]:3: error: use of undeclared identifier 'c' [CppInteropParseError]
  // CHECK:STDERR:    28 |   c.t = 123;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  c.t = 123;
}
''';

// --- fail_todo_generic_class_method_call.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C(T: type) {
  fn M(unused ref self) {}
}

inline Cpp '''
void F() {
  // CHECK:STDERR: fail_todo_generic_class_method_call.carbon:[[@LINE+7]]:18: error: implicit instantiation of undefined template 'Carbon::C<int>' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Carbon::C<int> c;
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR: fail_todo_generic_class_method_call.carbon:[[@LINE-9]]:18: note: template is declared here [CppInteropParseNote]
  // CHECK:STDERR:     4 | class C(T: type) {
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  Carbon::C<int> c;
  c.M();
}
''';
