// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/tuple_and_struct_type_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/tuple_and_struct_type_value.carbon

// --- tuple_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  let T:! (type, ) = ({}, );
  // Passes a symbolic binding of type TupleType.
  InterfaceParam(T);
  ConstraintParam(T);

  // Converts from a TupleLiteral with a symbolic type to a symbolic `type`.
  let U:! type = (T.0, );
  // Passes a symbolic `type`.
  InterfaceParam(U);
  ConstraintParam(U);

  let Empty:! () = ();
  InterfaceParam(Empty);
  ConstraintParam(Empty);
}

// --- struct_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  let T:! {} = {};
  // Passes a symbolic binding of type StructType.
  InterfaceParam(T);
  ConstraintParam(T);
}

// --- fail_struct_value_with_field_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  let T:! {.a: type} = {.a = ()};
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `{.a: type}` into type implementing `Z` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   InterfaceParam(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+7]]:3: note: type `{.a: type}` does not implement interface `Core.ImplicitAs(Z)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   InterfaceParam(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE-11]]:19: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn InterfaceParam(U:! Z) {}
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  InterfaceParam(T);
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `{.a: type}` into type implementing `N` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   ConstraintParam(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+7]]:3: note: type `{.a: type}` does not implement interface `Core.ImplicitAs(N)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ConstraintParam(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_value_with_field_to_facet_value.carbon:[[@LINE-21]]:20: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn ConstraintParam(U:! N) {}
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  ConstraintParam(T);
}
