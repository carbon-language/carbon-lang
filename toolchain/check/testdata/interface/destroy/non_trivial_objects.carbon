// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/non_trivial_objects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/non_trivial_objects.carbon

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

// Types used by the splits below.

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
class NonTrivial {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class TrivialBase {}

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
base class NonTrivialBase {
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

class ContainsNonTrivialMember {
  var n: NonTrivial;
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
}

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

class NestedNonTrivialMember {
  var inner: ContainsNonTrivialMember;
}

class DestroyImplAndTrivialBase {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class NonTrivialAndTrivialMembers {
  var n: NonTrivial;
  var a: i32;
}

class TrivialAndNonTrivialMembers {
  var a: i32;
  var n: NonTrivial;
}

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var a: i32;
}

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var n: NonTrivial;
}

class AdaptIntWithDestroyImpl {
  adapt i32;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AdaptNonTrivial {
  adapt NonTrivial;
}

class AdaptNonTrivialStruct {
  adapt {.n: NonTrivial};
}

class AdaptAdaptNonTrivial {
  adapt AdaptNonTrivial;
}

// `array(T, N)` is the class `Core.Array(T, N)`, so this is an adapter of an
// adapter.
class AdaptNonTrivialArray {
  adapt array(NonTrivial, 2);
}

// --- fail_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestArrayOfNonTrivial(x: array(NonTrivial, 3)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-31: error: cannot convert type `array(NonTrivial, 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArrayOfNonTrivial(x: array(array(NonTrivial, 2), 3)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-31: error: cannot convert type `array(array(NonTrivial, 2), 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestDestroyImpl(x: NonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestContainsNonTrivialMember(x: ContainsNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `ContainsNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromNonTrivialBase(x: DerivedFromNonTrivialBase) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DerivedFromNonTrivialBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-27]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNonTrivialGrandparent(x: NonTrivialGrandparent) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NonTrivialGrandparent` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-37]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNestedNonTrivialMember(x: NestedNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NestedNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-47]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDestroyImplAndTrivialBase(x: DestroyImplAndTrivialBase) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DestroyImplAndTrivialBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-57]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestGenericClassOfNonTrivial(x: DestroyableBox(NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DestroyableBox(NonTrivial as Core.Destroy)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-67]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNonTrivialAndTrivialMembers(x: NonTrivialAndTrivialMembers) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NonTrivialAndTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-77]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialAndNonTrivialMembers(x: TrivialAndNonTrivialMembers) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `TrivialAndNonTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-87]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNonTrivialBaseAndTrivialMember(x: NonTrivialBaseAndTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NonTrivialBaseAndTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-97]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialBaseAndNonTrivialMember(x: TrivialBaseAndNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `TrivialBaseAndNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-107]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestAdaptIntWithDestroyImpl(x: AdaptIntWithDestroyImpl) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `AdaptIntWithDestroyImpl` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-117]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: In each function below, `CheckHasTrivialDestruction` should fail.

fn TestAdaptNonTrivial(x: AdaptNonTrivial) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

fn TestAdaptNonTrivialStruct(x: AdaptNonTrivialStruct) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

fn TestAdaptAdaptNonTrivial(x: AdaptAdaptNonTrivial) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

fn TestAdaptNonTrivialArray(x: AdaptNonTrivialArray) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestConstNonTrivial(x: const NonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_const.carbon:[[@LINE+4]]:3-31: error: cannot convert type `const NonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_const.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestMaybeUnformedNonTrivial(x: Core.MaybeUnformed(NonTrivial)) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestStructWithNonTrivial(x: {.a: i32, .b: NonTrivial}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.a: i32, .b: NonTrivial}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNonTrivialFirst(x: {.a: NonTrivial, .b: i32}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.a: NonTrivial, .b: i32}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedNonTrivial(x: {.a: {.b: NonTrivial}}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.a: {.b: NonTrivial}}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-27]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestTupleWithNonTrivial(x: (i32, NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(i32, NonTrivial)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNonTrivialFirst(x: (NonTrivial, i32)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(NonTrivial, i32)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedNonTrivial(x: ((NonTrivial,),)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `((NonTrivial,),)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-27]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}
