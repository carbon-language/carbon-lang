// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy_format_per_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy_format_per_category.carbon

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

// Type-based versions of the functions above, for types that have no values in
// these tests: abstract, incomplete, `partial`, and symbolic types.
fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}

// --- types.carbon

library "[[@TEST_NAME]]";

// Types shared by the splits below.

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

// Has non-trivial destruction because of a member.
base class NonTrivialMemberBase {
  var n: NonTrivial;
}

abstract class Abstract {}

class Incomplete;

// A generic class with a member of type `T`. Tests that the destruction of a
// generic class comes from its type argument: `DestroyableBox(i32)` has trivial
// destruction, and `DestroyableBox(NonTrivial)` has non-trivial destruction.
// The generics tests also use it with a symbolic argument.
class DestroyableBox(T: Core.Destroy) {
  var t: T;
}

// An interface with an associated type that has no constraints. For a symbolic
// `T`, `T.U` is a type that isn't known to be destroyable.
interface HasUnconstrainedAssociatedType {
  let U: type;
}

// --- destroyable.h

using FunctionPointer = void (*)();
using Long = long;
using NullptrT = decltype(nullptr);

struct TrivialDestructor {};

struct DefaultedDestructor {
  ~DefaultedDestructor() = default;
};

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

// A user-provided destructor is non-trivial, even when its body is empty.
struct UserProvidedDestructor {
  ~UserProvidedDestructor() {}
};

struct HasTrivialMember {
  TrivialDestructor t;
  int i;
};

struct HasNonTrivialMember {
  NonTrivialDestructor n;
};

struct DerivedFromTrivial : TrivialDestructor {};

struct DerivedFromNonTrivial : NonTrivialDestructor {};

template <typename T>
struct Holder {
  T t;
};

struct VirtualFunction {
  virtual void F();
};

struct VirtualDestructor {
  virtual ~VirtualDestructor();
};

// --- no_destroy.h

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
};

struct ProtectedDestructor {
protected:
  ~ProtectedDestructor() = default;
};

struct PrivateDestructor {
private:
  ~PrivateDestructor() = default;
};

// The destructor is implicitly deleted, because the member's is deleted.
struct HasDeletedMember {
  DeletedDestructor d;
};

// --- unhandled.h

void Function();

template <typename T>
struct Template {};

// --- builtins.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestBool(x: bool) {
  CheckHasTrivialDestruction(x);
}

fn TestI32(x: i32) {
  CheckHasTrivialDestruction(x);
}

fn TestU8(x: u8) {
  CheckHasTrivialDestruction(x);
}

fn TestF64(x: f64) {
  CheckHasTrivialDestruction(x);
}

fn TestIntLiteral() {
  CheckHasTrivialDestruction(1);
}

fn TestFloatLiteral() {
  CheckHasTrivialDestruction(1.0);
}

fn TestCharLiteral() {
  CheckHasTrivialDestruction('a');
}

fn TestType() {
  CheckTypeHasTrivialDestruction(type);
}

fn TestFacetType() {
  CheckTypeHasTrivialDestruction(Core.Destroy);
}

fn TestForm() {
  CheckHasTrivialDestruction(form(var i32));
}

class C {
  fn Get(self) -> i32;
}

fn TestBoundMethod(c: C) {
  CheckHasTrivialDestruction(c.Get);
}

fn G() {}

fn TestFunction() {
  CheckHasTrivialDestruction(G);
}

fn TestFunctionWithSelf() {
  CheckHasTrivialDestruction((i32 as Core.Copy).Op);
}

fn TestPointer(x: i32*) {
  CheckHasTrivialDestruction(x);
}

// --- fail_builtins_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C {
  var x: i32;
}

fn TestUnboundElement() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<unbound element of class C>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(C.x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(C.x);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<unbound element of class C>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(C.x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-20]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(C.x);
}

interface I {
  fn F();
}

fn TestAssociatedEntity() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<associated entity in I>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(I.F);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-36]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(I.F);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<associated entity in I>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(I.F);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-46]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(I.F);
}

class Generic(T: type) {}

fn TestGenericClass() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Generic>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(Generic);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-60]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(Generic);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Generic>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Generic);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-70]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Generic);
}

interface GenericInterface(T: type) {}

fn TestGenericInterface() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(GenericInterface);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-84]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(GenericInterface);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericInterface);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-94]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericInterface);
}

constraint GenericConstraint(T: type) {}

fn TestGenericNamedConstraint() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(GenericConstraint);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-108]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(GenericConstraint);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericConstraint);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-118]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericConstraint);
}

namespace N;

fn TestNamespace() {
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<namespace>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(N);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-132]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(N);

  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<namespace>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(N);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_builtins_no_destroy.carbon:[[@LINE-142]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(N);
}

// --- arrays.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "no_destroy.h";

fn TestEmptyArray(x: array(i32, 0)) {
  CheckHasTrivialDestruction(x);
}

fn TestEmptyArrayOfNonTrivial(x: array(NonTrivial, 0)) {
  CheckHasTrivialDestruction(x);
}

fn TestEmptyArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 0)) {
  CheckHasTrivialDestruction(x);
}

fn TestArray(x: array(i32, 3)) {
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArray(x: array(array(i32, 2), 3)) {
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfEmptyArrayOfNonTrivial(x: array(array(NonTrivial, 0), 3)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_arrays_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestArrayOfNonTrivial(x: array(NonTrivial, 3)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_arrays_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `array(NonTrivial, 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArrayOfNonTrivial(x: array(array(NonTrivial, 2), 3)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_arrays_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `array(array(NonTrivial, 2), 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_non_trivial.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_arrays_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn TestArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 2)) {
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArrayOfDeletedDestructor(
    x: array(array(Cpp.DeletedDestructor, 2), 3)) {
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-42]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_arrays_no_destroy.carbon:[[@LINE-52]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- structs.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn TestEmptyStruct() {
  CheckHasTrivialDestruction({});
}
//@dump-sem-ir-end

fn TestStruct(x: {.a: i32, .b: bool}) {
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedTrivialStruct(x: {.a: {.b: i32}}) {
  CheckHasTrivialDestruction(x);
}

// --- fail_structs_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestStructWithNonTrivial(x: {.a: i32, .b: NonTrivial}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: i32, .b: NonTrivial}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNonTrivialFirst(x: {.a: NonTrivial, .b: i32}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: NonTrivial, .b: i32}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedNonTrivial(x: {.a: {.b: NonTrivial}}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: {.b: NonTrivial}}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_non_trivial.carbon:[[@LINE-37]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_structs_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn TestStructWithDeletedDestructor(x: {.d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedDeletedDestructor(x: {.a: {.d: Cpp.DeletedDestructor}}) {
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-41]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-51]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithTrivialAndDeletedDestructor(
    x: {.a: i32, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-64]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-74]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithDeletedDestructorAndTrivial(
    x: {.d: Cpp.DeletedDestructor, .a: i32}) {
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-87]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_structs_no_destroy.carbon:[[@LINE-97]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- tuples.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn TestEmptyTuple(x: ()) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

fn TestTuple(x: (i32, bool)) {
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedTrivialTuple(x: (i32, (bool, i32))) {
  CheckHasTrivialDestruction(x);
}

// --- fail_tuples_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestTupleWithNonTrivial(x: (i32, NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `(i32, NonTrivial)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNonTrivialFirst(x: (NonTrivial, i32)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `(NonTrivial, i32)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedNonTrivial(x: ((NonTrivial,),)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `((NonTrivial,),)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_non_trivial.carbon:[[@LINE-37]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_tuples_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn TestTupleWithDeletedDestructor(x: (Cpp.DeletedDestructor,)) {
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedDeletedDestructor(x: ((Cpp.DeletedDestructor,),)) {
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-41]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-51]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithTrivialAndDeletedDestructor(x: (i32, Cpp.DeletedDestructor)) {
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-63]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-73]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithDeletedDestructorAndTrivial(x: (Cpp.DeletedDestructor, i32)) {
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-85]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuples_no_destroy.carbon:[[@LINE-95]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

//@dump-sem-ir-begin
class EmptyClass {}

fn TestEmptyClass(x: EmptyClass) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

class TrivialFields {
  var a: i32;
  var b: bool;
}

fn TestTrivialFields(x: TrivialFields) {
  CheckHasTrivialDestruction(x);
}

class DerivedFromTrivialBase {
  extend base: TrivialBase;
  var a: i32;
}

fn TestDerivedFromTrivialBase(x: DerivedFromTrivialBase) {
  CheckHasTrivialDestruction(x);
}

class ContainsTrivialMember {
  var member: TrivialFields;
}

fn TestContainsTrivialMember(x: ContainsTrivialMember) {
  CheckHasTrivialDestruction(x);
}

fn TestGenericClassOfTrivial(x: DestroyableBox(i32)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_classes_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestDestroyImpl(x: NonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `NonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class ContainsNonTrivialMember {
  var n: NonTrivial;
}

fn TestContainsNonTrivialMember(x: ContainsNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `ContainsNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
}

fn TestDerivedFromNonTrivialBase(x: DerivedFromNonTrivialBase) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `DerivedFromNonTrivialBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-45]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn TestNonTrivialGrandparent(x: NonTrivialGrandparent) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `NonTrivialGrandparent` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-63]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class NestedNonTrivialMember {
  var inner: ContainsNonTrivialMember;
}

fn TestNestedNonTrivialMember(x: NestedNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `NestedNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-81]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class DestroyImplAndTrivialBase {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn TestDestroyImplAndTrivialBase(x: DestroyImplAndTrivialBase) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `DestroyImplAndTrivialBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-103]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestGenericClassOfNonTrivial(x: DestroyableBox(NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `DestroyableBox(NonTrivial as Core.SubobjectDestroy & Core.DestroyImpl)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-117]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class NonTrivialAndTrivialMembers {
  var n: NonTrivial;
  var a: i32;
}

fn TestNonTrivialAndTrivialMembers(x: NonTrivialAndTrivialMembers) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `NonTrivialAndTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-136]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class TrivialAndNonTrivialMembers {
  var a: i32;
  var n: NonTrivial;
}

fn TestTrivialAndNonTrivialMembers(x: TrivialAndNonTrivialMembers) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialAndNonTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-155]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var a: i32;
}

fn TestNonTrivialBaseAndTrivialMember(x: NonTrivialBaseAndTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `NonTrivialBaseAndTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-174]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var n: NonTrivial;
}

fn TestTrivialBaseAndNonTrivialMember(x: TrivialBaseAndNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialBaseAndNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_non_trivial.carbon:[[@LINE-193]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_classes_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "no_destroy.h";

fn TestAbstract() {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:27: error: cannot convert type `Abstract` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasDestruction(Abstract);
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:23:35: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasDestruction(Abstract);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:34: error: cannot convert type `Abstract` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasTrivialDestruction(Abstract);
  // CHECK:STDERR:                                  ^~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-18]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:24:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(Abstract);
}

fn TestIncomplete() {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:27: error: cannot convert type `Incomplete` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasDestruction(Incomplete);
  // CHECK:STDERR:                           ^~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-30]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:23:35: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasDestruction(Incomplete);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:34: error: cannot convert type `Incomplete` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasTrivialDestruction(Incomplete);
  // CHECK:STDERR:                                  ^~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-40]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:24:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(Incomplete);
}

class DeletedDestructorField {
  var d: Cpp.DeletedDestructor;
}

fn TestDeletedDestructorField(x: DeletedDestructorField) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    77 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-54]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-60]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorField` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-68]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorField` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-78]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class NestedDeletedDestructorMember {
  var inner: DeletedDestructorField;
}

fn TestNestedDeletedDestructorMember(x: NestedDeletedDestructorMember) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-94]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-104]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class DeletedDestructorAndTrivialMembers {
  var d: Cpp.DeletedDestructor;
  var a: i32;
}

fn TestDeletedDestructorAndTrivialMembers(
    x: DeletedDestructorAndTrivialMembers) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-122]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-132]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class TrivialAndDeletedDestructorMembers {
  var a: i32;
  var d: Cpp.DeletedDestructor;
}

fn TestTrivialAndDeletedDestructorMembers(
    x: TrivialAndDeletedDestructorMembers) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-150]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-160]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

base class DeletedDestructorFieldBase {
  var d: Cpp.DeletedDestructor;
}

class DeletedDestructorBaseAndTrivialMember {
  extend base: DeletedDestructorFieldBase;
  var a: i32;
}

fn TestDeletedDestructorBaseAndTrivialMember(
    x: DeletedDestructorBaseAndTrivialMember) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-182]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-192]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class TrivialBaseAndDeletedDestructorMember {
  extend base: TrivialBase;
  var d: Cpp.DeletedDestructor;
}

fn TestTrivialBaseAndDeletedDestructorMember(
    x: TrivialBaseAndDeletedDestructorMember) {
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-210]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_classes_no_destroy.carbon:[[@LINE-220]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_classes_derived_from_abstract.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class DerivedFromAbstract {
  extend base: Abstract;
}

// TODO: Support destroying classes that derive from abstract types.
fn TestDerivedFromAbstract(x: DerivedFromAbstract) {
  // CHECK:STDERR: fail_todo_classes_derived_from_abstract.carbon:[[@LINE+8]]:3: error: cannot convert type `DerivedFromAbstract` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_classes_derived_from_abstract.carbon:[[@LINE-12]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_dynamic_classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: Decide what kind of destruction a dynamic class has. A dynamic class
// has a virtual function, either directly or through a base class. Today these
// have trivial destruction.

base class VirtualFunction {
  virtual fn F(self);
}

fn TestVirtualFunction(x: VirtualFunction) {
  CheckHasTrivialDestruction(x);
}

class DerivedFromVirtualFunction {
  extend base: VirtualFunction;
}

fn TestDerivedFromVirtualFunction(x: DerivedFromVirtualFunction) {
  CheckHasTrivialDestruction(x);
}

// --- todo_optional.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: `Core.Optional` depends on `Core.MaybeUnformed` being correct, so this
// stays a TODO until `Core.MaybeUnformed` is complete.
fn TestOptional(x: Core.Optional(i32)) {
  CheckHasTrivialDestruction(x);
}

// --- adapters.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class AdaptInt {
  adapt i32;
}

fn TestAdaptInt(x: AdaptInt) {
  CheckHasTrivialDestruction(x);
}

class AdaptTrivialArray {
  adapt array(i32, 2);
}

fn TestAdaptTrivialArray(x: AdaptTrivialArray) {
  CheckHasTrivialDestruction(x);
}

// --- fail_adapters_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class AdaptIntWithDestroyImpl {
  adapt i32;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn TestAdaptIntWithDestroyImpl(x: AdaptIntWithDestroyImpl) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_adapters_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `AdaptIntWithDestroyImpl` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapters_non_trivial.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_adapters_of_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: In each function below, `CheckHasTrivialDestruction` should fail.

class AdaptNonTrivial {
  adapt NonTrivial;
}

fn TestAdaptNonTrivial(x: AdaptNonTrivial) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

class AdaptNonTrivialStruct {
  adapt {.n: NonTrivial};
}

fn TestAdaptNonTrivialStruct(x: AdaptNonTrivialStruct) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

class AdaptAdaptNonTrivial {
  adapt AdaptNonTrivial;
}

fn TestAdaptAdaptNonTrivial(x: AdaptAdaptNonTrivial) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// `array(T, N)` is the class `Core.Array(T, N)`, so this is an adapter of an
// adapter.
class AdaptNonTrivialArray {
  adapt array(NonTrivial, 2);
}

fn TestAdaptNonTrivialArray(x: AdaptNonTrivialArray) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_todo_adapters_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

class AdaptDeletedDestructor {
  adapt Cpp.DeletedDestructor;
}

// TODO: Both checks should fail. Today only the C++ deleted-destructor error is
// reported.
fn TestAdaptDeletedDestructor(x: AdaptDeletedDestructor) {
  // CHECK:STDERR: fail_todo_adapters_no_destroy.carbon:[[@LINE+12]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    26 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_todo_adapters_no_destroy.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_todo_adapters_no_destroy.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- wrappers.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestConst(x: const i32) {
  CheckHasTrivialDestruction(x);
}

fn TestPartialTrivialBase() {
  CheckTypeHasTrivialDestruction(partial TrivialBase);
}

fn TestPartialAbstract() {
  CheckTypeHasTrivialDestruction(partial Abstract);
}

// --- fail_wrappers_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestConstNonTrivial(x: const NonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_wrappers_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `const NonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrappers_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestPartialNonTrivialMemberBase() {
  CheckTypeHasDestruction(partial NonTrivialMemberBase);

  // CHECK:STDERR: fail_wrappers_non_trivial.carbon:[[@LINE+8]]:34: error: cannot convert type `partial NonTrivialMemberBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasTrivialDestruction(partial NonTrivialMemberBase);
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrappers_non_trivial.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:24:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(partial NonTrivialMemberBase);
}

// --- todo_fail_wrappers_of_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: `CheckTypeHasTrivialDestruction` should fail once `DestroyImpl.Op`
// takes `partial Self` instead of `Self`.
fn TestPartialNonTrivialBase() {
  CheckTypeHasDestruction(partial NonTrivialBase);

  CheckTypeHasTrivialDestruction(partial NonTrivialBase);
}

// --- todo_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: Implement destruction for `Core.MaybeUnformed`.
fn TestMaybeUnformed(x: Core.MaybeUnformed(i32)) {
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_maybe_unformed_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestMaybeUnformedNonTrivial(x: Core.MaybeUnformed(NonTrivial)) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- generics.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(T);
}

fn TestStructOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction({.a: T});
}

fn TestTupleOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction((T, i32));
}

fn TestArrayOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(array(T, 2));
}

fn TestGenericClassOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(DestroyableBox(T));
}

fn TestDestroyAndTrivialDestroy(generic T: Core.Destroy & Core.TrivialDestroy) {
  CheckTypeHasDestruction(T);

  CheckTypeHasTrivialDestruction(T);
}

// --- fail_generics_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestUnconstrained(generic T: type) {
  // CHECK:STDERR: fail_generics_no_destroy.carbon:[[@LINE+8]]:27: error: cannot convert type `T` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasDestruction(T);
  // CHECK:STDERR:                           ^
  // CHECK:STDERR: fail_generics_no_destroy.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:23:35: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasDestruction(T);
}

fn TestAssociatedType(generic T: HasUnconstrainedAssociatedType) {
  // CHECK:STDERR: fail_generics_no_destroy.carbon:[[@LINE+8]]:27: error: cannot convert type `T.(HasUnconstrainedAssociatedType.U)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasDestruction(T.U);
  // CHECK:STDERR:                           ^~~
  // CHECK:STDERR: fail_generics_no_destroy.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:23:35: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasDestruction(T.U);
}

// --- generics_trivial_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestTrivialDestroy(generic T: Core.TrivialDestroy) {
  CheckTypeHasDestruction(T);
}

// --- fail_todo_generics_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestUnconstrained(generic T: type) {
  // CHECK:STDERR: fail_todo_generics_trivial.carbon:[[@LINE+8]]:34: error: cannot convert type `T` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasTrivialDestruction(T);
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: fail_todo_generics_trivial.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:24:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(T);
}

fn TestAssociatedType(generic T: HasUnconstrainedAssociatedType) {
  // CHECK:STDERR: fail_todo_generics_trivial.carbon:[[@LINE+8]]:34: error: cannot convert type `T.(HasUnconstrainedAssociatedType.U)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckTypeHasTrivialDestruction(T.U);
  // CHECK:STDERR:                                  ^~~
  // CHECK:STDERR: fail_todo_generics_trivial.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:24:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(T.U);
}

// TODO: This should fail, because `T` might not have trivial destruction.
fn TestDestroy(generic T: Core.Destroy) {
  CheckTypeHasTrivialDestruction(T);
}

// --- cpp_builtins.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestFunctionPointer(x: Cpp.FunctionPointer) {
  CheckHasTrivialDestruction(x);
}

fn TestLong(x: Cpp.Long) {
  CheckHasTrivialDestruction(x);
}

fn TestNullptr(x: Cpp.NullptrT) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_builtins_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "unhandled.h";

fn TestCppTemplateName() {
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(Cpp.Template);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Template);

  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Template);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Template);
}

fn TestCppOverloadSet() {
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Function);

  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_builtins_no_destroy.carbon:[[@LINE-39]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Function);
}

// --- cpp_arrays.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestArrayOfTrivialDestructor(x: array(Cpp.TrivialDestructor, 2)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_arrays_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestArrayOfNonTrivialDestructor(x: array(Cpp.NonTrivialDestructor, 2)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_arrays_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `array(Cpp.NonTrivialDestructor, 2)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_arrays_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- cpp_structs.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestStructWithTrivialDestructor(x: {.t: Cpp.TrivialDestructor}) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_structs_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestStructWithNonTrivialDestructor(x: {.n: Cpp.NonTrivialDestructor}) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_structs_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `{.n: Cpp.NonTrivialDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_structs_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- cpp_tuples.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestTupleWithTrivialDestructor(x: (Cpp.TrivialDestructor,)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_tuples_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestTupleWithNonTrivialDestructor(x: (Cpp.NonTrivialDestructor,)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_tuples_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.NonTrivialDestructor,)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_tuples_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- cpp_classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "destroyable.h";

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

class CarbonWithTrivialDestructorMember {
  var t: Cpp.TrivialDestructor;
}

fn TestCarbonWithTrivialDestructorMember(x: CarbonWithTrivialDestructorMember) {
  CheckHasTrivialDestruction(x);
}

class CarbonDerivedFromTrivialDestructor {
  extend base: Cpp.TrivialDestructor;
}

fn TestCarbonDerivedFromTrivialDestructor(
    x: CarbonDerivedFromTrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

fn TestDestroyableBoxOfTrivialDestructor(
    x: DestroyableBox(Cpp.TrivialDestructor)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_classes_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "destroyable.h";

fn TestNonTrivialDestructor(x: Cpp.NonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.NonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestUserProvidedDestructor(x: Cpp.UserProvidedDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.UserProvidedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-24]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHasNonTrivialMember(x: Cpp.HasNonTrivialMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.HasNonTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-38]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromNonTrivial(x: Cpp.DerivedFromNonTrivial) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DerivedFromNonTrivial` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-52]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHolderOfNonTrivialDestructor(x: Cpp.Holder(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Holder` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-66]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHolderOfCarbonNonTrivial(x: Cpp.Holder(NonTrivial)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Holder` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-80]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class CarbonWithNonTrivialDestructorMember {
  var n: Cpp.NonTrivialDestructor;
}

fn TestCarbonWithNonTrivialDestructorMember(
    x: CarbonWithNonTrivialDestructorMember) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `CarbonWithNonTrivialDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-99]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

class CarbonDerivedFromNonTrivialDestructor {
  extend base: Cpp.NonTrivialDestructor;
}

fn TestCarbonDerivedFromNonTrivialDestructor(
    x: CarbonDerivedFromNonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `CarbonDerivedFromNonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-118]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDestroyableBoxOfNonTrivialDestructor(
    x: DestroyableBox(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `DestroyableBox(Cpp.NonTrivialDestructor as Core.SubobjectDestroy & Core.DestroyImpl)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_non_trivial.carbon:[[@LINE-133]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_classes_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn TestDeletedDestructor(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHasDeletedMember(x: Cpp.HasDeletedMember) {
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+24]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    66 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-40]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:18:21: note: destructor of 'HasDeletedMember' is implicitly deleted because field 'd' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR:    18 |   DeletedDestructor d;
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-44]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-49]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-57]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_classes_no_destroy.carbon:[[@LINE-67]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_cpp_inaccessible_destructors.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestProtectedDestructor(x: Cpp.ProtectedDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestPrivateDestructor(x: Cpp.PrivateDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- todo_cpp_dynamic_classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// TODO: Decide what kind of destruction a dynamic class has, as in
// `todo_dynamic_classes.carbon`. Today this has trivial destruction.
fn TestVirtualFunction(x: Cpp.VirtualFunction) {
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_cpp_dynamic_classes_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// TODO: implement Core.DynamicDestroy.
fn TestVirtualDestructor(x: Cpp.VirtualDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_cpp_dynamic_classes_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.VirtualDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_dynamic_classes_non_trivial.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- cpp_adapters.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

class AdaptTrivialDestructor {
  adapt Cpp.TrivialDestructor;
}

fn TestAdaptTrivialDestructor(x: AdaptTrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_cpp_adapters_of_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

class AdaptNonTrivialDestructor {
  adapt Cpp.NonTrivialDestructor;
}

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestAdaptNonTrivialDestructor(x: AdaptNonTrivialDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- cpp_wrappers.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestConstTrivialDestructor(x: const Cpp.TrivialDestructor) {
  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_wrappers_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

fn TestConstNonTrivialDestructor(x: const Cpp.NonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_wrappers_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `const Cpp.NonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_wrappers_non_trivial.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_cpp_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// TODO: Implement destruction for `Core.MaybeUnformed`.
fn TestMaybeUnformedTrivialDestructor(
    x: Core.MaybeUnformed(Cpp.TrivialDestructor)) {
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_cpp_maybe_unformed_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// TODO: `CheckHasTrivialDestruction` should fail.
fn TestMaybeUnformedNonTrivialDestructor(
    x: Core.MaybeUnformed(Cpp.NonTrivialDestructor)) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_cpp_carbon_cpp_non_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// A C++ type that holds a Carbon type that holds a C++ type.
class CarbonHoldsNonTrivialDestructor {
  var n: Cpp.NonTrivialDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsNonTrivialDestructor {
  Carbon::CarbonHoldsNonTrivialDestructor c;
};
''';

fn TestCppHoldsCarbonHoldsNonTrivialDestructor(
    x: Cpp.CppHoldsCarbonHoldsNonTrivialDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_cpp_carbon_cpp_non_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.CppHoldsCarbonHoldsNonTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_carbon_cpp_non_trivial.carbon:[[@LINE-21]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_cpp_carbon_cpp_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "destroyable.h";

// A C++ type that holds a Carbon type that holds a C++ type.
class CarbonHoldsTrivialDestructor {
  var t: Cpp.TrivialDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsTrivialDestructor {
  Carbon::CarbonHoldsTrivialDestructor c;
};
''';

// TODO: This should have trivial destruction, because every subobject does.
fn TestCppHoldsCarbonHoldsTrivialDestructor(
    x: Cpp.CppHoldsCarbonHoldsTrivialDestructor) {
  // CHECK:STDERR: fail_todo_cpp_carbon_cpp_trivial.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.CppHoldsCarbonHoldsTrivialDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_carbon_cpp_trivial.carbon:[[@LINE-20]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_cpp_carbon_cpp_no_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

// A C++ type that holds a Carbon type that holds a C++ type.
// CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE+26]]:36: error: attempt to use a deleted function [CppInteropParseError]
// CHECK:STDERR:    34 | class CarbonHoldsDeletedDestructor {
// CHECK:STDERR:       |                                    ^
// CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:33:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn SelfDestruct[T: Destroy](ref t: T) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE+15]]:1: error: cannot convert type `CarbonHoldsDeletedDestructor` into type implementing `Core.DestroyImpl & Core.SubobjectDestroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: class CarbonHoldsDeletedDestructor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:33:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn SelfDestruct[T: Destroy](ref t: T) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE+8]]:36: error: attempt to use a deleted function [CppInteropParseError]
// CHECK:STDERR:    34 | class CarbonHoldsDeletedDestructor {
// CHECK:STDERR:       |                                    ^
// CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE-24]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR:
class CarbonHoldsDeletedDestructor {
  var d: Cpp.DeletedDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsDeletedDestructor {
  Carbon::CarbonHoldsDeletedDestructor c;
};
''';

// TODO: `CheckHasDestruction` should fail too, because
// `CarbonHoldsDeletedDestructor` can't be destroyed. Today it passes; the only
// errors are at the definition of `CarbonHoldsDeletedDestructor` and from
// `CheckHasTrivialDestruction`.
fn TestCppHoldsCarbonHoldsDeletedDestructor(
    x: Cpp.CppHoldsCarbonHoldsDeletedDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.CppHoldsCarbonHoldsDeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_carbon_cpp_no_destroy.carbon:[[@LINE-51]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// CHECK:STDOUT: --- structs.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TestEmptyStruct.type: type = fn_type @TestEmptyStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestEmptyStruct: %TestEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.609: %TrivialDestroy.type = facet_value %empty_struct_type, (%custom_witness.447, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.30e: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.609) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyStruct.decl: %TestEmptyStruct.type = fn_decl @TestEmptyStruct [concrete = constants.%TestEmptyStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %.loc8_31.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.1: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %.loc8_32.1: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.1 [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.2: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %.loc8_32.2: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.2 [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.609) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.30e]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_31.2: %empty_struct_type = converted %.loc8_31.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%.loc8_31.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuples.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt.334: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.218: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt.334 [concrete]
// CHECK:STDOUT:   %TestEmptyTuple.type: type = fn_type @TestEmptyTuple [concrete]
// CHECK:STDOUT:   %TestEmptyTuple: %TestEmptyTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.3bb: %TrivialDestroy.type = facet_value %empty_tuple.type, (%custom_witness.447, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.674) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.6e1: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.3bb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyTuple.decl: %TestEmptyTuple.type = fn_decl @TestEmptyTuple [concrete = constants.%TestEmptyTuple] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt.334]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.218]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_23.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_23.3: type = converted %.loc7_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyTuple(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.1: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %.loc8_31.1: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.1 [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.2: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %.loc8_31.2: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.2 [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.3bb) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.6e1]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- classes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.dc9: type = pattern_type %EmptyClass [concrete]
// CHECK:STDOUT:   %x.param_patt.622: %pattern_type.dc9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.a86: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt.622 [concrete]
// CHECK:STDOUT:   %TestEmptyClass.type: type = fn_type @TestEmptyClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestEmptyClass: %TestEmptyClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_31.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c8: %SubobjectDestroy.type = facet_value %EmptyClass, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ec0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c8) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.d78: %TrivialDestroy.type = facet_value %EmptyClass, (%custom_witness.447, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.ec0) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.6bc: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.d78) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %EmptyClass.decl: type = class_decl @EmptyClass [concrete = constants.%EmptyClass] {} {}
// CHECK:STDOUT:   %TestEmptyClass.decl: %TestEmptyClass.type = fn_decl @TestEmptyClass [concrete = constants.%TestEmptyClass] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.dc9 = value_param_pattern [concrete = constants.%x.param_patt.622]
// CHECK:STDOUT:     %x.patt: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.a86]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %EmptyClass = value_param call_param0
// CHECK:STDOUT:     %EmptyClass.ref: type = name_ref EmptyClass, file.%EmptyClass.decl [concrete = constants.%EmptyClass]
// CHECK:STDOUT:     %x: %EmptyClass = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyClass {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyClass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyClass(%x.param: %EmptyClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %EmptyClass = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc11_31.1: %TrivialDestroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %.loc11_31.1: %TrivialDestroy.type = converted constants.%EmptyClass, %TrivialDestroy.facet.loc11_31.1 [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc11_31.2: %TrivialDestroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %.loc11_31.2: %TrivialDestroy.type = converted constants.%EmptyClass, %TrivialDestroy.facet.loc11_31.2 [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.d78) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.6bc]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_31.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_31.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
