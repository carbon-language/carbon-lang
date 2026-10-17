// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/cpp_objects_with_trivial_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/cpp_objects_with_trivial_destroy.carbon

// --- types.h

using FunctionPointer = void (*)();
using Long = long;
using NullptrT = decltype(nullptr);

struct TrivialDestructor {};

struct DefaultedDestructor {
  ~DefaultedDestructor() = default;
};

struct HasTrivialMember {
  TrivialDestructor t;
  int i;
};

struct DerivedFromTrivial : TrivialDestructor {};

template <typename T>
struct Holder {
  T t;
};

struct VirtualFunction {
  virtual void F();
};

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
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

// A generic class with a member of type `T`. Tests that the destruction of a
// generic class comes from its type argument: `DestroyableBox(i32)` has trivial
// destruction, and `DestroyableBox(NonTrivial)` has non-trivial destruction.
// The generics tests also use it with a symbolic argument.
class DestroyableBox(T: Core.Destroy) {
  var t: T;
}

class CarbonWithTrivialDestructorMember {
  var t: Cpp.TrivialDestructor;
}

class CarbonDerivedFromTrivialDestructor {
  extend base: Cpp.TrivialDestructor;
}

class AdaptTrivialDestructor {
  adapt Cpp.TrivialDestructor;
}

class CarbonHoldsTrivialDestructor {
  var t: Cpp.TrivialDestructor;
}

// --- cpp_function_pointer.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestFunctionPointer(x: Cpp.FunctionPointer) {
  CheckHasTrivialDestruction(x);
}

// --- int.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestLong(x: Cpp.Long) {
  CheckHasTrivialDestruction(x);
}

// --- array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestArrayOfTrivialDestructor(x: array(Cpp.TrivialDestructor, 2)) {
  CheckHasTrivialDestruction(x);
}

fn TestEmptyArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 0)) {
  CheckHasTrivialDestruction(x);
}

// --- class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

fn TestTrivialDestructor(x: Cpp.TrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

fn TestDefaultedDestructor(x: Cpp.DefaultedDestructor) {
  CheckHasTrivialDestruction(x);
}

fn TestHasTrivialMember(x: Cpp.HasTrivialMember) {
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromTrivial(x: Cpp.DerivedFromTrivial) {
  CheckHasTrivialDestruction(x);
}

fn TestHolderOfI32(x: Cpp.Holder(i32)) {
  CheckHasTrivialDestruction(x);
}

fn TestCarbonWithTrivialDestructorMember(x: CarbonWithTrivialDestructorMember) {
  CheckHasTrivialDestruction(x);
}

fn TestCarbonDerivedFromTrivialDestructor(
    x: CarbonDerivedFromTrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

fn TestDestroyableBoxOfTrivialDestructor(
    x: DestroyableBox(Cpp.TrivialDestructor)) {
  CheckHasTrivialDestruction(x);
}

fn TestAdaptTrivialDestructor(x: AdaptTrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

// --- todo_class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: Decide what kind of destruction a dynamic class has. Today this has
// trivial destruction.
fn TestVirtualFunction(x: Cpp.VirtualFunction) {
  CheckHasTrivialDestruction(x);
}

// --- todo_class_nullptr.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: `Cpp.NullptrT` depends on `Core.MaybeUnformed` being correct, so this
// stays a TODO until `Core.MaybeUnformed` is complete.
fn TestNullptr(x: Cpp.NullptrT) {
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_cpp_carbon_cpp.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

// A C++ type that holds a Carbon type that holds a C++ type.
inline Cpp '''
struct CppHoldsCarbonHoldsTrivialDestructor {
  Carbon::CarbonHoldsTrivialDestructor c;
};
''';

// TODO: This should have trivial destruction, because every subobject does.
fn TestCppHoldsCarbonHoldsTrivialDestructor(
    x: Cpp.CppHoldsCarbonHoldsTrivialDestructor) {
  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.CppHoldsCarbonHoldsTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestConstTrivialDestructor(x: const Cpp.TrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

// --- todo_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: Implement destruction for `Core.MaybeUnformed`.
fn TestMaybeUnformedTrivialDestructor(
    x: Core.MaybeUnformed(Cpp.TrivialDestructor)) {
  CheckHasTrivialDestruction(x);
}

// --- struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestStructWithTrivialDestructor(x: {.t: Cpp.TrivialDestructor}) {
  CheckHasTrivialDestruction(x);
}

// --- tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestTupleWithTrivialDestructor(x: (Cpp.TrivialDestructor,)) {
  CheckHasTrivialDestruction(x);
}
