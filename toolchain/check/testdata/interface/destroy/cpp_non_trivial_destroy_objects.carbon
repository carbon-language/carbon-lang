// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/cpp_non_trivial_destroy_objects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/cpp_non_trivial_destroy_objects.carbon

// --- types.h

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

// A user-provided destructor is non-trivial, even when its body is empty.
struct UserProvidedDestructor {
  ~UserProvidedDestructor() {}
};

struct HasNonTrivialMember {
  NonTrivialDestructor n;
};

struct DerivedFromNonTrivial : NonTrivialDestructor {};

template <typename T>
struct Holder {
  T t;
};

struct VirtualDestructor {
  virtual ~VirtualDestructor();
};

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.
//
// Use this function to confirm non-trivial destruction (pass) and types that
// can't be destroyed (fail).
fn CheckHasDestruction[T: Core.Destroy](_: T) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm trivial destruction (pass), non-trivial
// destruction (fail), and types that can't be destroyed (fail).
//
// `CheckHasTrivialDestruction` should only be combined with
// `CheckHasDestruction` for types that are expected to not have trivial
// destruction. Using both on a trivially destroyable type (e.g. `i32`) is
// redundant.
fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}

// --- types.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

// Types used by the splits below.

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
class NonTrivial {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// A generic class with a member of type `T`. Tests that the destruction of a
// generic class comes from its type argument: `DestroyableBox(i32)` has trivial
// destruction, and `DestroyableBox(NonTrivial)` has non-trivial destruction.
// The generics tests also use it with a symbolic argument.
class DestroyableBox(T: Core.Destroy) {
  var t: T;
}

class CarbonWithNonTrivialDestructorMember {
  var n: Cpp.NonTrivialDestructor;
}

class CarbonDerivedFromNonTrivialDestructor {
  extend base: Cpp.NonTrivialDestructor;
}

class CarbonHoldsNonTrivialDestructor {
  var n: Cpp.NonTrivialDestructor;
}

class AdaptNonTrivialDestructor {
  adapt Cpp.NonTrivialDestructor;
}

// --- fail_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestArrayOfNonTrivialDestructor(x: array(Cpp.NonTrivialDestructor, 2)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-31: error: cannot convert type `array(Cpp.NonTrivialDestructor, 2)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

fn TestNonTrivialDestructor(x: Cpp.NonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.NonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestUserProvidedDestructor(x: Cpp.UserProvidedDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.UserProvidedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHasNonTrivialMember(x: Cpp.HasNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.HasNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-28]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromNonTrivial(x: Cpp.DerivedFromNonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.DerivedFromNonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-38]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHolderOfNonTrivialDestructor(x: Cpp.Holder(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.Holder` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-48]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHolderOfCarbonNonTrivial(x: Cpp.Holder(NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.Holder` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-58]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestCarbonWithNonTrivialDestructorMember(
    x: CarbonWithNonTrivialDestructorMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `CarbonWithNonTrivialDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-69]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestCarbonDerivedFromNonTrivialDestructor(
    x: CarbonDerivedFromNonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `CarbonDerivedFromNonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-80]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDestroyableBoxOfNonTrivialDestructor(
    x: DestroyableBox(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DestroyableBox(Cpp.NonTrivialDestructor as Core.SubobjectDestroy & Core.DestroyImpl)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-91]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// A C++ type that holds a Carbon type that holds a C++ type.
inline Cpp '''
struct CppHoldsCarbonHoldsNonTrivialDestructor {
  Carbon::CarbonHoldsNonTrivialDestructor c;
};
''';

fn TestCppHoldsCarbonHoldsNonTrivialDestructor(
    x: Cpp.CppHoldsCarbonHoldsNonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.CppHoldsCarbonHoldsNonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-109]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestAdaptNonTrivialDestructor(x: AdaptNonTrivialDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: implement Core.DynamicDestroy.
fn TestVirtualDestructor(x: Cpp.VirtualDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_class_dynamic.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.VirtualDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_dynamic.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestConstNonTrivialDestructor(x: const Cpp.NonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_const.carbon:[[@LINE+4]]:3-31: error: cannot convert type `const Cpp.NonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_const.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestMaybeUnformedNonTrivialDestructor(
    x: Core.MaybeUnformed(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestStructWithNonTrivialDestructor(x: {.n: Cpp.NonTrivialDestructor}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.n: Cpp.NonTrivialDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestTupleWithNonTrivialDestructor(x: (Cpp.NonTrivialDestructor,)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(Cpp.NonTrivialDestructor,)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}
