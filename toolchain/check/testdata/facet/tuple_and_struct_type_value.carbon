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
impl forall [U: type] U as Z {}

constraint N {}

fn InterfaceParam(unused generic U: Z) {}
fn ConstraintParam(unused generic U: N) {}

fn F(generic T: (type, ), generic U: type, generic Empty: ()) {
  // Passes a symbolic binding of type TupleType.
  InterfaceParam(T);
  ConstraintParam(T);

  // Passes a symbolic `type`.
  InterfaceParam(U);
  ConstraintParam(U);

  InterfaceParam(Empty);
  ConstraintParam(Empty);
}

fn G(generic T: (type, )) {
  F(T, (T.0, ), ());
}

fn H() {
  G(({}, ));
}

// --- struct_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U: type] U as Z {}

constraint N {}

fn InterfaceParam(unused generic U: Z) {}
fn ConstraintParam(unused generic U: N) {}

fn F(generic T: {}) {
  // Passes a symbolic binding of type StructType.
  InterfaceParam(T);
  ConstraintParam(T);
}

fn G() {
  F({});
}

// --- fail_struct_value_with_field_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl forall [U: type] U as Z {}

constraint N {}

fn InterfaceParam(unused generic U: Z) {}
fn ConstraintParam(unused generic U: N) {}

fn F(generic T: {.a: type}) {
  // CHECK:STDERR:        ╭─┤ fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ fn InterfaceParam(unused generic U: Z) {}
  // CHECK:STDERR:        ·                                  ┬───
  // CHECK:STDERR:        ·                                  ╰──┤ initializing generic parameter `U`
  // CHECK:STDERR:        ·                                     │ declared here
  // CHECK:STDERR:        ·                                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   InterfaceParam(T);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `{.a: type}` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `Z`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `{.a: type}` does not implement interface `Core.ImplicitAs(Z)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  InterfaceParam(T);
  // CHECK:STDERR:        ╭─┤ fail_struct_value_with_field_to_facet_value.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ fn ConstraintParam(unused generic U: N) {}
  // CHECK:STDERR:        ·                                   ┬───
  // CHECK:STDERR:        ·                                   ╰──┤ initializing generic parameter `U`
  // CHECK:STDERR:        ·                                      │ declared here
  // CHECK:STDERR:        ·                                      │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   ConstraintParam(T);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `{.a: type}` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `N`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `{.a: type}` does not implement interface `Core.ImplicitAs(N)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  ConstraintParam(T);
}

fn G() {
  F({.a = ()});
}
