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

class C {
  fn M();
}

inline Cpp '''
void F() {
  Carbon::C::M();
}
''';

// --- method_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(self);
}

inline Cpp '''
void F() {
  Carbon::C c;
  c.M();
}
''';

// --- method_rvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(self);
}

inline Cpp '''
void F() {
  Carbon::C().M();
}
''';

// --- method_const_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(self);
}

inline Cpp '''
void F(const Carbon::C& c) {
  c.M();
}
''';

// --- ref_method_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(ref self);
}

inline Cpp '''
void F() {
  Carbon::C c;
  c.M();
}
''';

// --- fail_ref_method_rvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(ref self);
}

inline Cpp '''
void F() {
  // CHECK:STDERR: fail_ref_method_rvalue.carbon:[[@LINE+7]]:3: error: 'this' argument to member function 'M' is an rvalue, but function has non-const lvalue ref-qualifier [CppInteropParseError]
  // CHECK:STDERR:    17 |   Carbon::C().M();
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_ref_method_rvalue.carbon:[[@LINE-8]]:17: note: 'M' declared here [CppInteropParseNote]
  // CHECK:STDERR:     5 |   fn M(ref self);
  // CHECK:STDERR:       |                 ^
  // CHECK:STDERR:
  Carbon::C().M();
}
''';

// --- fail_ref_method_const_lvalue.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  fn M(ref self);
}

inline Cpp '''
void F(const Carbon::C& c) {
  // CHECK:STDERR: fail_ref_method_const_lvalue.carbon:[[@LINE+7]]:3: error: 'this' argument to member function 'M' has type 'const Carbon::C', but function is not marked const [CppInteropParseError]
  // CHECK:STDERR:    17 |   c.M();
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_ref_method_const_lvalue.carbon:[[@LINE-8]]:17: note: 'M' declared here [CppInteropParseNote]
  // CHECK:STDERR:     5 |   fn M(ref self);
  // CHECK:STDERR:       |                 ^
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
