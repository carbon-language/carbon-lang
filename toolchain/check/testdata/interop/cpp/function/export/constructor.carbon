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

// --- constructors.carbon
package Other;

class OneArg {
  fn OneArg(i: i32) -> Self;
}

class GenericConstructor {
  fn GenericConstructor[T: type](_: T) -> Self {
    return {};
  }
}

class GenericClass(T: type) {
  fn GenericClass(_: T) -> Self {
    return {};
  }
}

// --- call_constructor.carbon
library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void f() {
  Carbon::Other::OneArg a(42);
}
''';

// --- fail_todo_call_generic_constructor.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: constructors.carbon:8:3: error: semantics TODO: `support exporting generic member functions` [SemanticsTodo]
// CHECK:STDERR:   fn GenericConstructor[T: type](_: T) -> Self {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void f() {
  // CHECK:STDERR: fail_todo_call_generic_constructor.carbon:[[@LINE+13]]:37: error: no matching constructor for initialization of 'Carbon::Other::GenericConstructor' [CppInteropParseError]
  // CHECK:STDERR:    24 |   Carbon::Other::GenericConstructor b(0);
  // CHECK:STDERR:       |                                     ^ ~
  // CHECK:STDERR: constructors.carbon:7:26: note: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const GenericConstructor' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     7 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: constructors.carbon:7:26: note: candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'GenericConstructor' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     7 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: constructors.carbon:7:26: note: candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     7 | class GenericConstructor {
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR:
  Carbon::Other::GenericConstructor b(0);
  // Carbon::Other::GenericClass<int> c(0);
}
''';

// --- fail_todo_call_generic_class_constructor.carbon
library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void f() {
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+12]]:18: error: no member named 'GenericClass' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    18 |   Carbon::Other::GenericClass<int> c(0);
  // CHECK:STDERR:       |                  ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+8]]:34: error: expected '(' for function-style cast or type construction [CppInteropParseError]
  // CHECK:STDERR:    18 |   Carbon::Other::GenericClass<int> c(0);
  // CHECK:STDERR:       |                               ~~~^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_generic_class_constructor.carbon:[[@LINE+4]]:36: error: use of undeclared identifier 'c' [CppInteropParseError]
  // CHECK:STDERR:    18 |   Carbon::Other::GenericClass<int> c(0);
  // CHECK:STDERR:       |                                    ^
  // CHECK:STDERR:
  Carbon::Other::GenericClass<int> c(0);
}
''';
