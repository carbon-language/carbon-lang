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

class TwoIntFields {
  var x: i32;
  var y: i32;
}

inline Cpp '''
void AccessTwoFields() {
  Carbon::TwoIntFields a;
  a.x = 12;
  a.y = 34;
}
''';

// --- field_inheritence.carbon
library "[[@TEST_NAME]]";
import Cpp;

base class BaseWithField {
  var x: i32;
}

class DerivedWithField {
  extend base: BaseWithField;
  var y: i32;
}

inline Cpp '''
void AccessInheritedField() {
  Carbon::DerivedWithField b;
  b.x = 12;
  b.y = 34;
}
''';

// --- class_field.carbon
library "[[@TEST_NAME]]";
import Cpp;

class ClassField {}

class ClassWithClassField {
  var c: ClassField;
}

inline Cpp '''
void g() {
  // In order for this to work, we need both class definitions to be exported to
  // Clang.
  Carbon::ClassWithClassField c;
}
''';

// --- fail_nonexistent_field.carbon
library "[[@TEST_NAME]]";
import Cpp;

class EmptyClass {}

inline Cpp '''
void AccessNonexistentField() {
  Carbon::EmptyClass a;
  // CHECK:STDERR: fail_nonexistent_field.carbon:[[@LINE+2]]:5: error: no member named 'x' in 'Carbon::EmptyClass' [CppInteropParseError]
  // CHECK:STDERR:
  a.x = 12;
  // CHECK:STDERR: fail_nonexistent_field.carbon:[[@LINE+2]]:5: error: no member named 'y' in 'Carbon::EmptyClass' [CppInteropParseError]
  // CHECK:STDERR:
  a.y = 34;
}
''';

// --- fail_unsupported_field_type.carbon
library "[[@TEST_NAME]]";
import Cpp;

class UnsupportedFieldType {
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+2]]:7-11: error: semantics TODO: `failed to map Carbon type to C++` [SemanticsTodo]
  // CHECK:STDERR:
  var x: ();
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+2]]:7-11: error: semantics TODO: `failed to map Carbon type to C++` [SemanticsTodo]
  // CHECK:STDERR:
  var y: ();
}

inline Cpp '''
void AccessUnsupportedField() {
  Carbon::UnsupportedFieldType a;
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+2]]:5: error: no member named 'x' in 'Carbon::UnsupportedFieldType' [CppInteropParseError]
  // CHECK:STDERR:
  a.x;
  // CHECK:STDERR: fail_unsupported_field_type.carbon:[[@LINE+2]]:5: error: no member named 'y' in 'Carbon::UnsupportedFieldType' [CppInteropParseError]
  // CHECK:STDERR:
  a.y;
}
''';

// --- fail_adapter.carbon
library "[[@TEST_NAME]]";
import Cpp;

class AdaptedClass {
  adapt ();
}

inline Cpp '''
void AccessAdaptedField() {
  Carbon::AdaptedClass a;
  // CHECK:STDERR: fail_adapter.carbon:[[@LINE+2]]:5: error: no member named 'x' in 'Carbon::AdaptedClass' [CppInteropParseError]
  // CHECK:STDERR:
  a.x = 12;
}
''';

// --- fail_invalid.carbon
library "[[@TEST_NAME]]";
import Cpp;

// This should not cause a crash.
class InvalidBaseClass {
  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+2]]:16-21: error: name `Whoops` not found [NameNotFound]
  // CHECK:STDERR:
  extend base: Whoops;
}

inline Cpp '''
Carbon::InvalidBaseClass invalid_x;
''';

// --- fail_private_field.carbon
library "[[@TEST_NAME]]";
import Cpp;

class ClassWithPrivateField {
  fn Make() -> ClassWithPrivateField {
    return {.x = 123};
  }

  private var x: i32;
}

inline Cpp '''
int AccessPrivateField() {
  auto c = Carbon::ClassWithPrivateField::Make();
  // CHECK:STDERR: fail_private_field.carbon:[[@LINE+3]]:12: error: 'x' is a private member of 'Carbon::ClassWithPrivateField' [CppInteropParseError]
  // CHECK:STDERR: fail_private_field.carbon:[[@LINE-7]]:16: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:
  return c.x;
}
''';
