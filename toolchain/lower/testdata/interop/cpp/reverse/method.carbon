// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/reverse/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/reverse/method.carbon

// --- method.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  // CHECK:STDERR: method.carbon:[[@LINE+4]]:3: error: semantics TODO: `unsupported: C++ calling a Carbon function with an implicit parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn F[self: Self]() { self; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self]() { self; }
}

inline Cpp '''
void CallF() {
  // CHECK:STDERR: method.carbon:[[@LINE+4]]:15: error: no member named 'F' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    19 |   Carbon::A().F();
  // CHECK:STDERR:       |   ~~~~~~~~~~~ ^
  // CHECK:STDERR:
  Carbon::A().F();
}
''';

// --- method_with_args_and_return.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  // CHECK:STDERR: method_with_args_and_return.carbon:[[@LINE+4]]:3: error: semantics TODO: `unsupported: C++ calling a Carbon function with an implicit parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn F[self: Self](a: i32) -> i32 {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](a: i32) -> i32 {
    self;
    return a;
  }
}

inline Cpp '''
int CallF() {
  // CHECK:STDERR: method_with_args_and_return.carbon:[[@LINE+4]]:22: error: no member named 'F' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    22 |   return Carbon::A().F(123);
  // CHECK:STDERR:       |          ~~~~~~~~~~~ ^
  // CHECK:STDERR:
  return Carbon::A().F(123);
}
''';

fn CallCallF() -> i32 { return Cpp.CallF(); }

// --- static.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  fn F() {}
}

inline Cpp '''
void CallF() {
  Carbon::A::F();
}
''';

fn CallCallF() { Cpp.CallF(); }

