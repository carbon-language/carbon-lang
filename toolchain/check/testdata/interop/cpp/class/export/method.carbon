// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/method.carbon

// --- static_method.carbon
library "[[@TEST_NAME]]";
import Cpp;

class StaticMethodClass {
  fn M();
}

inline Cpp '''
void StaticMethod() {
  Carbon::StaticMethodClass::M();
}
''';

// --- method_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class MethodLValueClass {
  fn M(self);
}

inline Cpp '''
void MethodLValue() {
  Carbon::MethodLValueClass c;
  c.M();
}
''';

// --- method_rvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class MethodRValueClass {
  fn M(self);
}

inline Cpp '''
void MethodRValue() {
  Carbon::MethodRValueClass().M();
}
''';

// --- method_const_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class MethodConstLValueClass {
  fn M(self);
}

inline Cpp '''
void MethodConstLValue(const Carbon::MethodConstLValueClass& c) {
  c.M();
}
''';

// --- ref_method_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class RefMethodLValueClass {
  fn M(ref self);
}

inline Cpp '''
void RefMethodLValue() {
  Carbon::RefMethodLValueClass c;
  c.M();
}
''';

// --- fail_ref_method_rvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class RefMethodRValueClass {
  fn M(ref self);
}

inline Cpp '''
void RefMethodRValue() {
  // CHECK:STDERR: fail_ref_method_rvalue.carbon:[[@LINE+3]]:3-8: error: 'this' argument to member function 'M' is an rvalue, but function has non-const lvalue ref-qualifier [CppInteropParseError]
  // CHECK:STDERR: fail_ref_method_rvalue.carbon:[[@LINE-6]]:17: note: 'M' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::RefMethodRValueClass().M();
}
''';

// --- fail_ref_method_const_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class RefMethodConstLValueClass {
  fn M(ref self);
}

inline Cpp '''
void RefMethodConstLValue(const Carbon::RefMethodConstLValueClass& c) {
  // CHECK:STDERR: fail_ref_method_const_lvalue.carbon:[[@LINE+3]]:3: error: 'this' argument to member function 'M' has type 'const Carbon::RefMethodConstLValueClass', but function is not marked const [CppInteropParseError]
  // CHECK:STDERR: fail_ref_method_const_lvalue.carbon:[[@LINE-6]]:17: note: 'M' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  c.M();
}
''';

// --- method_alias.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {
  fn F();
}

alias G = A.F;

inline Cpp '''
void call() { Carbon::G(); }
''';
