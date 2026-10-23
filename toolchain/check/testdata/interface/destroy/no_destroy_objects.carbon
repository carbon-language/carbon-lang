// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/no_destroy_objects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/no_destroy_objects.carbon

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

interface I {
  fn F();
}

interface GenericInterface(T: type) {}

constraint GenericConstraint(T: type) {}

class Generic(T: type) {}

class C {
  var x: i32;
}

// --- fail_associated_entity.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestAssociatedEntity() {
  // CHECK:STDERR: fail_associated_entity.carbon:[[@LINE+4]]:3-26: error: cannot convert type `<associated entity in I>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_generic_interface.carbon:[[@LINE+4]]:3-39: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_generic_named_constraint.carbon:[[@LINE+4]]:3-40: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_namespace.carbon:[[@LINE+4]]:3-24: error: cannot convert type `<namespace>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_parameterized_class.carbon:[[@LINE+4]]:3-30: error: cannot convert type `<type of Generic>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_unbound_element.carbon:[[@LINE+4]]:3-26: error: cannot convert type `<unbound element of class C>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
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
