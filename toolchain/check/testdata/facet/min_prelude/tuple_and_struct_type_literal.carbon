// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/tuple_and_struct_type_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/tuple_and_struct_type_literal.carbon

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
