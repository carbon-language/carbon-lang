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

// --- fail_declare_after_poisoning.carbon
library "[[@TEST_NAME]]";

import Cpp;

namespace N;

inline Cpp '''
namespace Fallback {
  int *LaterFn();
  int *LaterNestedFn();
}
using namespace Fallback;
using namespace Carbon;
using namespace Carbon::N;

// Unqualified lookups search `Carbon` and `Carbon::N` (failing there without
// triggering typo correction) and resolve to `Fallback`. Following the normal
// Carbon name poisoning rules, these then trigger poisoning errors when those
// names are defined in Carbon.
// CHECK:STDERR: fail_declare_after_poisoning.carbon:[[@LINE+1]]:10: error: name `LaterFn` used before it was declared [NameUseBeforeDecl]
int *a = LaterFn();
int *b = LaterNestedFn();
''';

// CHECK:STDERR: fail_declare_after_poisoning.carbon:[[@LINE+3]]:4-10: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_declare_after_poisoning.carbon:[[@LINE-5]]:10: error: name `LaterNestedFn` used before it was declared [NameUseBeforeDecl]
fn LaterFn() -> i32;
// CHECK:STDERR: fail_declare_after_poisoning.carbon:[[@LINE+2]]:6-18: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N.LaterNestedFn() -> i32;

inline Cpp '''
// Subsequent qualified lookups find the newly declared Carbon functions rather
// than seeing cached negative lookup results.
int c = Carbon::LaterFn();
int d = Carbon::N::LaterNestedFn();
''';

// --- fail_declare_after_poisoning_with_lookup_failure.carbon
library "[[@TEST_NAME]]";

import Cpp;

namespace Nested;
fn Nested.EarlyFn() {}

inline Cpp '''
void TestLookupFailureInNested() {
  // TODO: Only produce one diagnostic here. The first one comes from Clang; the
  // second is a Carbon error for declaring a poisoned name. Normally we would
  // suppress the first error when we produce the second, but we don't do that
  // when the first error is produced by Clang.
  // CHECK:STDERR: fail_declare_after_poisoning_with_lookup_failure.carbon:[[@LINE+3]]:19-31: error: no member named 'LaterInNested' in namespace 'Carbon::Main::Nested' [CppInteropParseError]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_declare_after_poisoning_with_lookup_failure.carbon:[[@LINE+1]]:19: error: name `LaterInNested` used before it was declared [NameUseBeforeDecl]
  Carbon::Nested::LaterInNested();
  Carbon::Nested::EarlyFn();
}
''';

// CHECK:STDERR: fail_declare_after_poisoning_with_lookup_failure.carbon:[[@LINE+2]]:11-23: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn Nested.LaterInNested() {}

inline Cpp '''
void TestAfterTypoInNested() {
  // No additional error here.
  Carbon::Nested::LaterInNested();
}
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
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+4]]:11: error: no type named 'ExportedTyp' in namespace 'Carbon'; did you mean 'ExportedType'? [CppInteropParseError]
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+3]]:11-21: note: replace with `ExportedType` [CppInteropFixItReplacement]
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE-12]]:20: note: 'ExportedType' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::ExportedTyp* p_typo = p1;

  // When correcting a typo, Clang attempts to fully populate the name lookup
  // table for the context (namespace `Carbon` in this case). Ensure that
  // subsequent name lookups still work properly.
  Carbon::EarlyFn();
  Carbon::ExportedType* p2 = p1;
  // CHECK:STDERR: fail_typo_in_carbon_namespace.carbon:[[@LINE+2]]:11-21: error: no member named 'NotDeclared' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::NotDeclared();
}
''';

class LaterType {
  var y: i32;
}

inline Cpp '''
void TestAfterTypoInCarbon() {
  static_assert(sizeof(Carbon::ExportedType) == 4);
  static_assert(sizeof(Carbon::LaterType) == 4);
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
  // TODO: It'd be nice if Clang's typo correction could correct this.
  // CHECK:STDERR: fail_typo_in_carbon_class.carbon:[[@LINE+2]]:20-31: error: no member named 'StaticMetho1' in 'Carbon::Main::MyClass' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::MyClass::StaticMetho1();

  // Later name lookups into the class should still work.
  Carbon::MyClass::StaticMethod1();
  Carbon::MyClass::StaticMethod2();
  Carbon::MyClass c{1};
  c.Method();
  static_assert(sizeof(Carbon::MyClass) == 4);

  // Typo correction does work once we've exported the name.
  // CHECK:STDERR: fail_typo_in_carbon_class.carbon:[[@LINE+4]]:20-31: error: no member named 'StaticMetho1' in 'Carbon::Main::MyClass'; did you mean 'StaticMethod1'? [CppInteropParseError]
  // CHECK:STDERR: fail_typo_in_carbon_class.carbon:[[@LINE+3]]:20-31: note: replace with `StaticMethod1` [CppInteropFixItReplacement]
  // CHECK:STDERR: fail_typo_in_carbon_class.carbon:[[@LINE-22]]:21: note: 'StaticMethod1' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::MyClass::StaticMetho1();
}
''';
