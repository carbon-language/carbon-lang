// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/tuple_and_struct_type_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/tuple_and_struct_type_literal.carbon

// --- tuple_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl () as Y {}

// The tuple literal converts implicitly to a type, and then to a facet value
// satisfying `Y`.
impl forall [T:! type] T as Z where .X = () {}

// --- complex_tuple_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl ((), {}) as Y {}

// The tuple literal converts implicitly to a type, and then to a facet value
// satisfying `Y`.
impl forall [T:! type] T as Z where .X = ((), {}) {}

// --- fail_mismatch_tuple_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl ((), {}) as Y {}

// CHECK:STDERR: fail_mismatch_tuple_literal_to_facet_value.carbon:[[@LINE+4]]:42: error: cannot convert type `({}, {})` into type implementing `Y` [ConversionFailureTypeToFacet]
// CHECK:STDERR: impl forall [T:! type] T as Z where .X = ({}, {}) {}
// CHECK:STDERR:                                          ^~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z where .X = ({}, {}) {}

// --- struct_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl {} as Y {}

// The struct literal converts implicitly to a type, and then to a facet value
// satisfying `Y`.
impl forall [T:! type] T as Z where .X = {} {}

// --- complex_struct_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl {.a: (), .b: {}} as Y {}

// The struct literal converts implicitly to a type, and then to a facet value
// satisfying `Y`.
impl forall [T:! type] T as Z where .X = {.a: (), .b: {}} {}

// --- fail_mismatch_struct_literal_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let X:! Y;
}

impl {.a: (), .b: {}} as Y {}

// CHECK:STDERR: fail_mismatch_struct_literal_to_facet_value.carbon:[[@LINE+4]]:42: error: cannot convert type `{.a: {}, .b: {}}` into type implementing `Y` [ConversionFailureTypeToFacet]
// CHECK:STDERR: impl forall [T:! type] T as Z where .X = {.a: {}, .b: {}} {}
// CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z where .X = {.a: {}, .b: {}} {}

// --- struct_literal_to_facet_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  InterfaceParam({});
  ConstraintParam({});
}

// --- fail_struct_literal_with_field_to_facet_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `{.a: ()}` into type implementing `Z` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   InterfaceParam({.a = ()});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE+7]]:3: note: type `{.a: ()}` does not implement interface `Core.ImplicitAs(Z)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   InterfaceParam({.a = ()});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE-10]]:19: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn InterfaceParam(U:! Z) {}
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  InterfaceParam({.a = ()});
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `{.a: ()}` into type implementing `N` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   ConstraintParam({.a = ()});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE+7]]:3: note: type `{.a: ()}` does not implement interface `Core.ImplicitAs(N)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ConstraintParam({.a = ()});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_literal_with_field_to_facet_type_parameter.carbon:[[@LINE-20]]:20: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn ConstraintParam(U:! N) {}
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  ConstraintParam({.a = ()});
}

// --- tuple_literal_to_facet_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U:! type] U as Z {}

constraint N {}

fn InterfaceParam(U:! Z) {}
fn ConstraintParam(U:! N) {}

fn F() {
  InterfaceParam(());
  ConstraintParam(());

  InterfaceParam(({}, ));
  ConstraintParam(({}, ));

  class C;
  InterfaceParam(({}, C));
  ConstraintParam(({}, C));
}

// --- struct_literal_to_generic_facet_type_parameter_with_concrete_specific.carbon
library "[[@TEST_NAME]]";

interface Z(U:! type) {}
impl forall [T:! type, U:! type] U as Z(T) {}

constraint N(U:! type) {}

class C;
fn InterfaceParam(U:! Z(C)) {}
fn ConstraintParam(U:! N(C)) {}

fn F() {
  InterfaceParam({});
  ConstraintParam({});
}

// --- tuple_literal_to_generic_facet_type_parameter_with_concrete_specific.carbon
library "[[@TEST_NAME]]";

interface Z(U:! type) {}
impl forall [T:! type, U:! type] U as Z(T) {}

constraint N(U:! type) {}

class C;
fn InterfaceParam(U:! Z(C)) {}
fn ConstraintParam(U:! N(C)) {}

fn F() {
  InterfaceParam(());
  ConstraintParam(());

  InterfaceParam(({}, ));
  ConstraintParam(({}, ));

  class C;
  InterfaceParam(({}, C));
  ConstraintParam(({}, C));
}

// --- struct_literal_to_generic_facet_type_parameter_with_symbolic_specific.carbon
library "[[@TEST_NAME]]";

interface Z(U:! type) {}
impl forall [T:! type, U:! type] U as Z(T) {}

constraint N(U:! type) {}

fn InterfaceParam(T:! type, U:! Z(T)) {}
fn ConstraintParam(T:! type, U:! N(T)) {}

fn F(T:! type) {
  InterfaceParam((), {});
  ConstraintParam((), {});

  InterfaceParam(T, {});
  ConstraintParam(T, {});
}

// --- tuple_literal_to_generic_facet_type_parameter_with_symbolic_specific.carbon
library "[[@TEST_NAME]]";

interface Z(U:! type) {}
impl forall [T:! type, U:! type] U as Z(T) {}

constraint N(U:! type) {}

fn InterfaceParam(T:! type, U:! Z(T)) {}
fn ConstraintParam(T:! type, U:! N(T)) {}

fn F(T:! type) {
  InterfaceParam({}, ());
  ConstraintParam({}, ());

  InterfaceParam(T, ());
  ConstraintParam(T, ());

  InterfaceParam(T, ({}, ));
  ConstraintParam(T, ({}, ));

  class C;
  InterfaceParam(T, ({}, C));
  ConstraintParam(T, ({}, C));
}
