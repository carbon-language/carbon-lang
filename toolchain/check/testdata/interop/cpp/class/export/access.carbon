// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/access.carbon

// --- fail_access_exported_private_var.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {}

class B {
  fn Make() -> Self { return {}; }
  private var a: A = {};
}

inline Cpp '''
void F() {
  auto b = Carbon::B::Make();
  // CHECK:STDERR: fail_access_exported_private_var.carbon:[[@LINE+5]]:5: error: 'a' is a private member of 'Carbon::Main::B' [CppInteropParseError]
  // CHECK:STDERR: fail_access_exported_private_var.carbon:[[@LINE-7]]:16: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_access_exported_private_var.carbon:[[@LINE+2]]:5: warning: expression result unused [CppInteropParseWarning]
  // CHECK:STDERR:
  b.a;
}
''';

// --- fail_access_exported_private_method.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {}

class B {
  fn Make() -> Self { return {}; }
  private fn F() {}
}

inline Cpp '''
void G() {
  auto b = Carbon::B::Make();
  // CHECK:STDERR: fail_access_exported_private_method.carbon:[[@LINE+3]]:5: error: 'F' is a private member of 'Carbon::Main::B' [CppInteropParseError]
  // CHECK:STDERR: fail_access_exported_private_method.carbon:[[@LINE-7]]:18: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:
  b.F();
}
''';

// --- fail_access_exported_private_var_in_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {}

class B(T: type) {
  private var a: A = {};
}

fn MakeB() -> B(A) {
  return {};
}

inline Cpp '''
void F() {
  auto b = Carbon::MakeB();
  // CHECK:STDERR: fail_access_exported_private_var_in_generic.carbon:[[@LINE+5]]:5: error: 'a' is a private member of 'Carbon::Main::B<Carbon::Main::A>' [CppInteropParseError]
  // CHECK:STDERR: fail_access_exported_private_var_in_generic.carbon:[[@LINE-11]]:16: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_access_exported_private_var_in_generic.carbon:[[@LINE+2]]:5: warning: expression result unused [CppInteropParseWarning]
  // CHECK:STDERR:
  b.a;
}
''';

// --- fail_todo_access_exported_private_method_in_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {}

class B(T: type) {
  // TODO: exporting of methods in generics is not yet supported.
  // CHECK:STDERR: fail_todo_access_exported_private_method_in_generic.carbon:[[@LINE+2]]:3-18: error: semantics TODO: `non-class non-namespace name scope` [SemanticsTodo]
  // CHECK:STDERR:
  private fn F() {}
}

fn MakeB() -> B(A) {
  return {};
}

inline Cpp '''
void F() {
  auto b = Carbon::MakeB();
  // TODO: this error should be that `F` is a private member of `B`.
  // CHECK:STDERR: fail_todo_access_exported_private_method_in_generic.carbon:[[@LINE+2]]:5: error: no member named 'F' in 'Carbon::Main::B<Carbon::Main::A>' [CppInteropParseError]
  // CHECK:STDERR:
  b.F();
}
''';
