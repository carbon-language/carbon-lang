// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/field.carbon

// --- field.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {
  var x: i32;
  var y: i32;
}

inline Cpp '''
void F() {
  Carbon::A a;
  a.x = 12;
  a.y = 34;
}
''';

// --- field_inheritence.carbon
library "[[@TEST_NAME]]";
import Cpp;

base class A {
  var x: i32;
}

class B {
  extend base: A;
  var y: i32;
}

inline Cpp '''
void F() {
  Carbon::B b;
  b.x = 12;
  b.y = 34;
}
''';

// --- fail_nonexistent_field.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {}

inline Cpp '''
void F() {
  Carbon::A a;
  // CHECK:STDERR: fail_nonexistent_field.carbon:[[@LINE+4]]:5: error: no member named 'x' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    13 |   a.x = 12;
  // CHECK:STDERR:       |   ~ ^
  // CHECK:STDERR:
  a.x = 12;
  // CHECK:STDERR: fail_nonexistent_field.carbon:[[@LINE+4]]:5: error: no member named 'y' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    18 |   a.y = 34;
  // CHECK:STDERR:       |   ~ ^
  // CHECK:STDERR:
  a.y = 34;
}
''';

// --- fail_unsupported_field_type.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `failed to map Carbon type to C++` [SemanticsTodo]
  // CHECK:STDERR:   var x: ();
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  var x: ();
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `failed to map Carbon type to C++` [SemanticsTodo]
  // CHECK:STDERR:   var y: ();
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  var y: ();
}

inline Cpp '''
void F() {
  Carbon::A a;
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+4]]:5: error: no member named 'x' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    24 |   a.x;
  // CHECK:STDERR:       |   ~ ^
  // CHECK:STDERR:
  a.x;
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+4]]:5: error: no member named 'y' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    29 |   a.y;
  // CHECK:STDERR:       |   ~ ^
  // CHECK:STDERR:
  a.y;
}
''';

// --- fail_adapter.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A {
  adapt ();
}

inline Cpp '''
void F() {
  Carbon::A a;
  // CHECK:STDERR: fail_adapter.carbon:[[@LINE+4]]:5: error: no member named 'x' in 'Carbon::A' [CppInteropParseError]
  // CHECK:STDERR:    15 |   a.x = 12;
  // CHECK:STDERR:       |   ~ ^
  // CHECK:STDERR:
  a.x = 12;
}
''';

// --- fail_invalid.carbon
library "[[@TEST_NAME]]";
import Cpp;

// This should not cause a crash.
class A {
  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:16: error: name `Whoops` not found [NameNotFound]
  // CHECK:STDERR:   extend base: Whoops;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  extend base: Whoops;
}

inline Cpp '''
Carbon::A x;
''';
