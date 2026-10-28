// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/no_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/no_destroy.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.
fn CheckHasDestruction(unused T: Core.Destroy) {}

// --- fail_associated_entity.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

interface I {
  fn F();
}

fn TestAssociatedEntity() {
  // CHECK:STDERR: fail_associated_entity.carbon:[[@LINE+5]]:23-25: error: cannot implicitly convert non-type value of type `<associated entity in I>` into type implementing `Core.Destroy` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_associated_entity.carbon:[[@LINE+4]]:23-25: note: type `<associated entity in I>` does not implement interface `Core.ImplicitAs(Core.Destroy)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: imported from: fail_associated_entity.carbon:[[@LINE-9]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(I.F);
}

// --- fail_generic_interface.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

interface GenericInterface(T: type) {}

fn TestGenericInterface() {
  // CHECK:STDERR: fail_generic_interface.carbon:[[@LINE+5]]:23-38: error: cannot implicitly convert non-type value of type `<type of GenericInterface>` into type implementing `Core.Destroy` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_generic_interface.carbon:[[@LINE+4]]:23-38: note: type `<type of GenericInterface>` does not implement interface `Core.ImplicitAs(Core.Destroy)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: imported from: fail_generic_interface.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(GenericInterface);
}

// --- fail_generic_named_constraint.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

constraint GenericConstraint(T: type) {}

fn TestGenericNamedConstraint() {
  // CHECK:STDERR: fail_generic_named_constraint.carbon:[[@LINE+5]]:23-39: error: cannot implicitly convert non-type value of type `<type of GenericConstraint>` into type implementing `Core.Destroy` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_generic_named_constraint.carbon:[[@LINE+4]]:23-39: note: type `<type of GenericConstraint>` does not implement interface `Core.ImplicitAs(Core.Destroy)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: imported from: fail_generic_named_constraint.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(GenericConstraint);
}

// --- fail_namespace.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

namespace N;

fn TestNamspace() {
  // CHECK:STDERR: fail_namespace.carbon:[[@LINE+4]]:23-31: error: cannot convert type `<namespace>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_namespace.carbon:[[@LINE-6]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(typeof(N));
}

// --- fail_parameterized_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C(T: type) {}

fn TestGenericClass() {
  // CHECK:STDERR: fail_parameterized_class.carbon:[[@LINE+5]]:23: error: cannot implicitly convert non-type value of type `<type of C>` into type implementing `Core.Destroy` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_parameterized_class.carbon:[[@LINE+4]]:23: note: type `<type of C>` does not implement interface `Core.ImplicitAs(Core.Destroy)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: imported from: fail_parameterized_class.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(C);
}

// --- fail_unbound_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C {
  var x: i32;
}

fn TestUnboundElement() {
  // CHECK:STDERR: fail_unbound_element.carbon:[[@LINE+4]]:23-33: error: cannot convert type `<unbound element of class C>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_unbound_element.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(typeof(C.x));
}

// --- fail_call.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// A function that returns a type. A call to it with a symbolic argument
// can't be resolved until the argument is known.
eval fn Identity(generic T: type) -> type { return T; }

// A generic class with a member whose type is a call to `Identity`.
class IdentityBox(T: type) {
  var t: Identity(T);
}

// `IdentityBox(T)` is used instead of `Identity(T)`, because a check that
// uses `Identity(T)` directly waits until `T` is known.
fn TestCall(generic T: Core.Destroy) {
  // CHECK:STDERR: fail_call.carbon:[[@LINE+4]]:23-36: error: cannot convert type `IdentityBox(T)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_call.carbon:[[@LINE-15]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(IdentityBox(T));
}

// --- fail_impl_witness_access.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// An interface with an associated type that has no constraints. For a symbolic
// `T`, `T.U` is a type that isn't known to be destroyable.
interface HasUnconstrainedAssociatedType {
  let U: type;
}

fn TestAssociatedType(generic T: HasUnconstrainedAssociatedType) {
  // CHECK:STDERR: fail_impl_witness_access.carbon:[[@LINE+4]]:23-25: error: cannot convert type `T.(HasUnconstrainedAssociatedType.U)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_impl_witness_access.carbon:[[@LINE-10]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(T.U);
}

// --- fail_symbolic_binding.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestUnconstrained(generic T: type) {
  // CHECK:STDERR: fail_symbolic_binding.carbon:[[@LINE+4]]:23: error: cannot convert type `T` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_symbolic_binding.carbon:[[@LINE-4]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(T);
}

// --- fail_tuple_access.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestTupleAccess(generic T: (type, type)) {
  // CHECK:STDERR: fail_tuple_access.carbon:[[@LINE+4]]:23-25: error: cannot convert type `<cannot stringify inst55000083: {kind: TupleAccess, arg0: inst55000022, arg1: element0, type: type(TypeType)}>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple_access.carbon:[[@LINE-4]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(T.0);
}

// --- fail_incomplete.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class Incomplete;

fn TestIncomplete() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:23-32: error: cannot convert type `Incomplete` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_incomplete.carbon:[[@LINE-6]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(Incomplete);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

abstract class Abstract {}

fn TestAbstract() {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:23-30: error: cannot convert type `Abstract` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-6]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:31-45: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckHasDestruction(Abstract);
}
