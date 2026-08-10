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

class GenericClass(T: type) {
  var t: T;
}

inline Cpp '''
void FGeneric() {
  Carbon::GenericClass<int> c;
  c.t = 123;
}
''';

// --- specific_alias.carbon
library "[[@TEST_NAME]]";
import Cpp;

class SpecificClass(T: type) {
  var t: T;
}
alias AliasInt = SpecificClass(i32);

inline Cpp '''
void FSpecific() {
  Carbon::AliasInt a;
  a.t = 123;
}
''';

// --- fail_unsupported_specific_arg.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_unsupported_specific_arg.carbon:[[@LINE+2]]:1-33: error: semantics TODO: `failed to map specific type arg to C++` [SemanticsTodo]
// CHECK:STDERR:
class UnsupportedClass(T: type) {
  var t: T;
}
alias AliasEmptyTuple = UnsupportedClass(());

inline Cpp '''
void FUnsupported() {
  // CHECK:STDERR: fail_unsupported_specific_arg.carbon:[[@LINE+4]]:11: error: semantics TODO: `interop with unsupported type` [SemanticsTodo]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unsupported_specific_arg.carbon:[[@LINE+2]]:11-25: error: no type named 'AliasEmptyTuple' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::AliasEmptyTuple a;
  a.t = 123;
}
''';

// --- fail_todo_generic_class_method_call.carbon
library "[[@TEST_NAME]]";
import Cpp;

class ClassWithMethod(T: type) {
  // CHECK:STDERR: fail_todo_generic_class_method_call.carbon:[[@LINE+2]]:3-25: error: semantics TODO: `non-class non-namespace name scope` [SemanticsTodo]
  // CHECK:STDERR:
  fn M(unused ref self) {}
}

inline Cpp '''
void FMethodCall() {
  Carbon::ClassWithMethod<int> c;
  // CHECK:STDERR: fail_todo_generic_class_method_call.carbon:[[@LINE+2]]:5: error: no member named 'M' in 'Carbon::ClassWithMethod<int>' [CppInteropParseError]
  // CHECK:STDERR:
  c.M();
}
''';
