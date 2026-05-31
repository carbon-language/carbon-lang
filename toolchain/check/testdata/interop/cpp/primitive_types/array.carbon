// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/array.carbon

// --- import.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
int arr[5];

struct X {
  int arr[5];
} x;
''';

fn F(n: i32) -> i32 {
  return Cpp.arr[n];
}

fn G(x: Cpp.X, n: i32) -> i32 {
  return x.arr[n];
}

// --- fail_todo_export_variable.carbon
library "[[@TEST_NAME]]";

import Cpp;

var arr: array(i32, 5);

inline Cpp '''
int F(int n) {
  // CHECK:STDERR: fail_todo_export_variable.carbon:[[@LINE+4]]:18: error: no member named 'arr' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    13 |   return Carbon::arr[n];
  // CHECK:STDERR:       |                  ^~~
  // CHECK:STDERR:
  return Carbon::arr[n];
}
''';

// --- export.carbon
library "[[@TEST_NAME]]";

import Cpp;

class X {
  var arr: array(i32, 5);
}

inline Cpp '''
int G(const Carbon::X& x, int n) {
  return x.arr[n];
}
''';

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:1: error: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class C;

fn F() -> array(C, 5)*;

inline Cpp '''
void G() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:33: error: subscript of pointer to incomplete type 'Carbon::C' [CppInteropParseError]
  // CHECK:STDERR:    19 |   Carbon::C *p = &(*Carbon::F())[0];
  // CHECK:STDERR:       |                   ~~~~~~~~~~~~~~^
  // CHECK:STDERR:
  Carbon::C *p = &(*Carbon::F())[0];
}
''';
