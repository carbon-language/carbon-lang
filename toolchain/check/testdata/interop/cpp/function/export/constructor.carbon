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
void CallOneArg() {
  Carbon::OneArg a(42);
}
''';

// --- fail_factory_is_method.carbon
library "[[@TEST_NAME]]";

import Cpp;

class FactoryIsMethod {
  fn FactoryIsMethod(self, i: i32) -> Self;
}

inline Cpp '''
void CallFactoryIsMethod() {
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE+13]]:27: error: no matching constructor for initialization of 'Carbon::FactoryIsMethod' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::FactoryIsMethod a(42);
  // CHECK:STDERR:       |                           ^ ~~
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-9]]:23: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const FactoryIsMethod' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryIsMethod {
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-12]]:23: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'FactoryIsMethod' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryIsMethod {
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR: fail_factory_is_method.carbon:[[@LINE-15]]:23: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryIsMethod {
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR:
  Carbon::FactoryIsMethod a(42);
}
''';

// --- fail_factory_doesnt_return_self.carbon
library "[[@TEST_NAME]]";

import Cpp;

class FactoryDoesntReturnSelf {
  fn FactoryDoesntReturnSelf(i: i32) -> i32;
}

inline Cpp '''
void CallFactoryDoesntReturnSelf() {
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE+13]]:35: error: no matching constructor for initialization of 'Carbon::FactoryDoesntReturnSelf' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::FactoryDoesntReturnSelf a(42);
  // CHECK:STDERR:       |                                   ^ ~~
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-9]]:31: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const FactoryDoesntReturnSelf' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryDoesntReturnSelf {
  // CHECK:STDERR:       |                               ^
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-12]]:31: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'FactoryDoesntReturnSelf' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryDoesntReturnSelf {
  // CHECK:STDERR:       |                               ^
  // CHECK:STDERR: fail_factory_doesnt_return_self.carbon:[[@LINE-15]]:31: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryDoesntReturnSelf {
  // CHECK:STDERR:       |                               ^
  // CHECK:STDERR:
  Carbon::FactoryDoesntReturnSelf a(42);
}
''';

// --- fail_factory_returns_by_ref.carbon
library "[[@TEST_NAME]]";

import Cpp;

class FactoryReturnsByRef {
  fn FactoryReturnsByRef(i: i32) -> ref Self;
}

inline Cpp '''
void CallFactoryReturnsByRef() {
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE+13]]:31: error: no matching constructor for initialization of 'Carbon::FactoryReturnsByRef' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::FactoryReturnsByRef a(42);
  // CHECK:STDERR:       |                               ^ ~~
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-9]]:27: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const FactoryReturnsByRef' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryReturnsByRef {
  // CHECK:STDERR:       |                           ^
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-12]]:27: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'FactoryReturnsByRef' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryReturnsByRef {
  // CHECK:STDERR:       |                           ^
  // CHECK:STDERR: fail_factory_returns_by_ref.carbon:[[@LINE-15]]:27: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class FactoryReturnsByRef {
  // CHECK:STDERR:       |                           ^
  // CHECK:STDERR:
  Carbon::FactoryReturnsByRef a(42);
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
void CallGenericConstructor() {
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
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+4]]:3: error: semantics TODO: `non-class non-namespace name scope` [SemanticsTodo]
  // CHECK:STDERR:   fn GenericClass(_: T) -> Self {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn GenericClass(_: T) -> Self {
    return {};
  }
}

inline Cpp '''
void CallGenericClass() {
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+13]]:29: error: no matching constructor for initialization of 'Carbon::GenericClass<int>' [CppInteropParseError]
  // CHECK:STDERR:    30 |   Carbon::GenericClass<int> c(0);
  // CHECK:STDERR:       |                             ^ ~
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE-15]]:29: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const GenericClass<int>' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericClass(T: type) {
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE-18]]:29: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'GenericClass<int>' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericClass(T: type) {
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE-21]]:29: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | class GenericClass(T: type) {
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR:
  Carbon::GenericClass<int> c(0);
}
''';
