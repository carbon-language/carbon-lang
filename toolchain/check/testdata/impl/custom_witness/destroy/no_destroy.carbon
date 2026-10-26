// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/no_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/no_destroy.carbon

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

// Identifies whether a type can be destroyed.
//
// Use this function to confirm non-trivial destruction (pass) and types that
// can't be destroyed (fail).
fn CheckTypeHasDestruction(unused T: Core.Destroy) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm trivial destruction (pass), non-trivial
// destruction (fail), and types that can't be destroyed (fail).
//
// `CheckTypeHasTrivialDestruction` should only be combined with
// `CheckTypeHasDestruction` for types that are expected to not have trivial
// destruction. Using both on a trivially destroyable type (e.g. `i32`) is
// redundant.
fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}

// --- types.carbon

library "[[@TEST_NAME]]";

// Types used by the splits below.

interface I {
  fn F();
}

interface GenericInterface(T: type) {}

constraint GenericConstraint(T: type) {}

class Generic(T: type) {}

class C {
  var x: i32;
}

abstract class Abstract {}

class Incomplete;

// An interface with an associated type that has no constraints. For a symbolic
// `T`, `T.U` is a type that isn't known to be destroyable.
interface HasUnconstrainedAssociatedType {
  let U: type;
}

// A function that returns a type. A call to it with a symbolic argument
// can't be resolved until the argument is known.
eval fn Identity(generic T: type) -> type { return T; }

// A generic class with a member whose type is a call to `Identity`.
class IdentityBox(T: type) {
  var t: Identity(T);
}

// --- fail_associated_entity.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestAssociatedEntity() {
  // CHECK:STDERR: fail_associated_entity.carbon:[[@LINE+4]]:3-26: error: cannot convert type `<associated entity in I>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_associated_entity.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(I.F);

  // CHECK:STDERR: fail_associated_entity.carbon:[[@LINE+4]]:3-33: error: cannot convert type `<associated entity in I>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_associated_entity.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(I.F);
}

// --- fail_generic_interface.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestGenericInterface() {
  // CHECK:STDERR: fail_generic_interface.carbon:[[@LINE+4]]:3-39: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_generic_interface.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(GenericInterface);

  // CHECK:STDERR: fail_generic_interface.carbon:[[@LINE+4]]:3-46: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_generic_interface.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericInterface);
}

// --- fail_generic_named_constraint.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestGenericNamedConstraint() {
  // CHECK:STDERR: fail_generic_named_constraint.carbon:[[@LINE+4]]:3-40: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_generic_named_constraint.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(GenericConstraint);

  // CHECK:STDERR: fail_generic_named_constraint.carbon:[[@LINE+4]]:3-47: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_generic_named_constraint.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericConstraint);
}

// --- fail_namespace.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

namespace N;

fn TestNamespace() {
  // CHECK:STDERR: fail_namespace.carbon:[[@LINE+4]]:3-24: error: cannot convert type `<namespace>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_namespace.carbon:[[@LINE-6]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(N);

  // CHECK:STDERR: fail_namespace.carbon:[[@LINE+4]]:3-31: error: cannot convert type `<namespace>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_namespace.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(N);
}

// --- fail_parameterized_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestGenericClass() {
  // CHECK:STDERR: fail_parameterized_class.carbon:[[@LINE+4]]:3-30: error: cannot convert type `<type of Generic>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_parameterized_class.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(Generic);

  // CHECK:STDERR: fail_parameterized_class.carbon:[[@LINE+4]]:3-37: error: cannot convert type `<type of Generic>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_parameterized_class.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Generic);
}

// --- fail_unbound_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestUnboundElement() {
  // CHECK:STDERR: fail_unbound_element.carbon:[[@LINE+4]]:3-26: error: cannot convert type `<unbound element of class C>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_unbound_element.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(C.x);

  // CHECK:STDERR: fail_unbound_element.carbon:[[@LINE+4]]:3-33: error: cannot convert type `<unbound element of class C>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_unbound_element.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(C.x);
}

// --- fail_call.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// `IdentityBox(T)` is used instead of `Identity(T)`, because a check that
// uses `Identity(T)` directly waits until `T` is known.
fn TestCall(generic T: Core.Destroy) {
  // CHECK:STDERR: fail_call.carbon:[[@LINE+4]]:27-40: error: cannot convert type `IdentityBox(T)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_call.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(IdentityBox(T));

  // CHECK:STDERR: fail_call.carbon:[[@LINE+4]]:34-47: error: cannot convert type `IdentityBox(T)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_call.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(IdentityBox(T));
}

// --- fail_impl_witness_access.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestAssociatedType(generic T: HasUnconstrainedAssociatedType) {
  // CHECK:STDERR: fail_impl_witness_access.carbon:[[@LINE+4]]:27-29: error: cannot convert type `T.(HasUnconstrainedAssociatedType.U)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_impl_witness_access.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(T.U);

  // CHECK:STDERR: fail_impl_witness_access.carbon:[[@LINE+4]]:34-36: error: cannot convert type `T.(HasUnconstrainedAssociatedType.U)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_impl_witness_access.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(T.U);
}

// --- fail_symbolic_binding.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestUnconstrained(generic T: type) {
  // CHECK:STDERR: fail_symbolic_binding.carbon:[[@LINE+4]]:27: error: cannot convert type `T` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_symbolic_binding.carbon:[[@LINE-4]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(T);

  // CHECK:STDERR: fail_symbolic_binding.carbon:[[@LINE+4]]:34: error: cannot convert type `T` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_symbolic_binding.carbon:[[@LINE-10]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(T);
}

// --- fail_tuple_access.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestTupleAccess(generic T: (type, type)) {
  // CHECK:STDERR: fail_tuple_access.carbon:[[@LINE+4]]:27-29: error: cannot convert type `<cannot stringify inst75000086: {kind: TupleAccess, arg0: inst75000025, arg1: element0, type: type(TypeType)}>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple_access.carbon:[[@LINE-4]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(T.0);

  // CHECK:STDERR: fail_tuple_access.carbon:[[@LINE+4]]:34-36: error: cannot convert type `<cannot stringify inst75000086: {kind: TupleAccess, arg0: inst75000025, arg1: element0, type: type(TypeType)}>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple_access.carbon:[[@LINE-10]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(T.0);
}

// --- fail_incomplete.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestIncomplete() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:27-36: error: cannot convert type `Incomplete` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_incomplete.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(Incomplete);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:34-43: error: cannot convert type `Incomplete` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_incomplete.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(Incomplete);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestAbstract() {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:27-34: error: cannot convert type `Abstract` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:25:35-49: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasDestruction(Abstract);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:34-41: error: cannot convert type `Abstract` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:36:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(Abstract);
}
