// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/constructor.carbon

// --- call_constructor.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OneArg {
  fn OneArg(i: i32) -> Self;
}

inline Cpp '''
void f() {
  Carbon::OneArg a(42);
}
''';

// --- fail_factory_is_method.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OneArg {
  fn OneArg(self, i: i32) -> Self;
}

inline Cpp '''
void f() {
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE+13]]:18: error: no matching constructor for initialization of 'Carbon::OneArg' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::OneArg a(42);
  // CHECK:STDERR:       |                  ^ ~~
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-9]]:14: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-12]]:14: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-15]]:14: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR:
  Carbon::OneArg a(42);
}
''';

// --- fail_factory_doesnt_return_self.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OneArg {
  fn OneArg(i: i32) -> i32;
}

inline Cpp '''
void f() {
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE+13]]:18: error: no matching constructor for initialization of 'Carbon::OneArg' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::OneArg a(42);
  // CHECK:STDERR:       |                  ^ ~~
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-9]]:14: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-12]]:14: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-15]]:14: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR:
  Carbon::OneArg a(42);
}
''';

// --- fail_factory_returns_by_ref.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OneArg {
  fn OneArg(i: i32) -> ref Self;
}

inline Cpp '''
void f() {
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE+13]]:18: error: no matching constructor for initialization of 'Carbon::OneArg' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::OneArg a(42);
  // CHECK:STDERR:       |                  ^ ~~
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-9]]:14: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-12]]:14: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'OneArg' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-15]]:14: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class OneArg {
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR:
  Carbon::OneArg a(42);
}
''';

// --- fail_todo_call_generic_constructor.carbon
library "[[@TEST_NAME]]";

import Cpp;

class GenericConstructor {
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE+4]]:3: error: semantics TODO: `support exporting generic member functions` [SemanticsTodo]
  // CHECK:STDERR:   fn GenericConstructor[T: type](_: T) -> Self {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn GenericConstructor[T: type](_: T) -> Self {
    return {};
  }
}

inline Cpp '''
void f() {
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE+13]]:30: error: no matching constructor for initialization of 'Carbon::GenericConstructor' [CppInteropParseError]
  // CHECK:STDERR:    30 |   Carbon::GenericConstructor b(0);
  // CHECK:STDERR:       |                              ^ ~
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE-15]]:26: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const GenericConstructor' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE-18]]:26: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'GenericConstructor' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE-21]]:26: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR:
  Carbon::GenericConstructor b(0);
}
''';

// --- fail_todo_call_generic_class_constructor.carbon
library "[[@TEST_NAME]]";

import Cpp;

class GenericClass(T: type) {
  fn GenericClass(_: T) -> Self {
    return {};
  }
}

inline Cpp '''
void f() {
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+7]]:29: error: implicit instantiation of undefined template 'Carbon::GenericClass<int>' [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::GenericClass<int> c(0);
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE-11]]:29: note: template is declared here [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericClass(T: type) {
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR:
  Carbon::GenericClass<int> c(0);
}
''';
