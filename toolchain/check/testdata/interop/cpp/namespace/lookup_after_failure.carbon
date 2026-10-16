// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace/lookup_after_failure.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace/lookup_after_failure.carbon

// --- fail_negative_lookup_via_using_directive.carbon
library "[[@TEST_NAME]]";

import Cpp;

namespace N;

inline Cpp '''
namespace Fallback {
  int LaterFn() { return 1; }
  int LaterNestedFn() { return 2; }
}
using namespace Fallback;
using namespace Carbon;
using namespace Carbon::N;

// Unqualified lookups search `Carbon` and `Carbon::N` (failing there without
// triggering typo correction) and resolve to `Fallback`.
// CHECK:STDERR: fail_negative_lookup_via_using_directive.carbon:[[@LINE+1]]:9: error: name `LaterFn` used before it was declared [NameUseBeforeDecl]
int a = LaterFn();
int b = LaterNestedFn();
''';

// CHECK:STDERR: fail_negative_lookup_via_using_directive.carbon:[[@LINE+3]]:4-10: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_lookup_via_using_directive.carbon:[[@LINE-5]]:9: error: name `LaterNestedFn` used before it was declared [NameUseBeforeDecl]
fn LaterFn() -> i32 { return 3; }
// CHECK:STDERR: fail_negative_lookup_via_using_directive.carbon:[[@LINE+2]]:6-18: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N.LaterNestedFn() -> i32 { return 4; }

inline Cpp '''
// Subsequent qualified lookups find the newly declared Carbon functions rather
// than seeing cached negative lookup results.
int c = Carbon::LaterFn();
int d = Carbon::N::LaterNestedFn();
''';

// --- fail_typo_in_carbon_namespace.carbon
library "[[@TEST_NAME]]";

import Cpp;

class ExportedType {
  var x: i32;
}

fn EarlyFn() {}

inline Cpp '''
Carbon::ExportedType* p1 = nullptr;

void TestTypoInCarbon() {
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+2]]:11-20: error: no member named 'NoSuchDecl' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::NoSuchDecl();
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+4]]:11: error: no type named 'ExportedTyp' in namespace 'Carbon'; did you mean 'ExportedType'? [CppInteropParseError]
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+3]]:11-21: note: replace with `ExportedType` [CppInteropFixItReplacement]
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE-15]]:20: note: 'ExportedType' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::ExportedTyp* p_typo = p1;
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+3]]:11-23: error: no member named 'LaterDeclared' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+1]]:11: error: name `LaterDeclared` used before it was declared [NameUseBeforeDecl]
  Carbon::LaterDeclared();
  Carbon::EarlyFn();
  Carbon::ExportedType* p2 = p1;
}
''';

// CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+2]]:4-16: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn LaterDeclared() -> i32 { return 42; }

class LaterType {
  var y: i32;
}

inline Cpp '''
void TestAfterTypoInCarbon() {
  int v = Carbon::LaterDeclared();
  static_assert(sizeof(Carbon::ExportedType) == 4);
  static_assert(sizeof(Carbon::LaterType) == 4);
}
''';

// --- fail_typo_in_nested_namespace.carbon
library "[[@TEST_NAME]]";

import Cpp;

namespace Nested;
fn Nested.EarlyFn() {}

inline Cpp '''
void TestTypoInNested() {
  // CHECK:STDERR: fail_typo_in_nested_namespace.carbon:[[@LINE+3]]:19-31: error: no member named 'LaterInNested' in namespace 'Carbon::Main::Nested' [CppInteropParseError]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_typo_in_nested_namespace.carbon:[[@LINE+1]]:19: error: name `LaterInNested` used before it was declared [NameUseBeforeDecl]
  Carbon::Nested::LaterInNested();
  Carbon::Nested::EarlyFn();
}
''';

// CHECK:STDERR: fail_typo_in_nested_namespace.carbon:[[@LINE+2]]:11-23: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn Nested.LaterInNested() {}

inline Cpp '''
void TestAfterTypoInNested() {
  Carbon::Nested::LaterInNested();
}
''';

// --- fail_typo_in_carbon_class.carbon
library "[[@TEST_NAME]]";

import Cpp;

class MyClass {
  var a: i32;
  fn StaticMethod1();
  fn StaticMethod2();
  fn Method(self);
}

inline Cpp '''
void TestTypoInClass() {
  Carbon::MyClass::StaticMethod1();
  // CHECK:STDERR: fail_typo_in_carbon_class.carbon:[[@LINE+2]]:20-31: error: no member named 'NoSuchMember' in 'Carbon::Main::MyClass' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::MyClass::NoSuchMember();
  Carbon::MyClass::StaticMethod1();
  Carbon::MyClass::StaticMethod2();
  Carbon::MyClass c{1};
  c.Method();
  static_assert(sizeof(Carbon::MyClass) == 4);
}
''';
