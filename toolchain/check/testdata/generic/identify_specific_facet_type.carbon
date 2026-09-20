// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/identify_specific_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/identify_specific_facet_type.carbon

// --- fail_error_in_identifying_require_facet_type.carbon
library "[[@TEST_NAME]]";

// Identification tries to form the specific interface `K(array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T: type) {}
constraint L(N: i32) {
  require impls K(array(i32, N));
}
constraint J(N: i32) {
  require impls L(N);
}

// CHECK:STDERR: fail_error_in_identifying_require_facet_type.carbon:[[@LINE+3]]:1-18: error: facet type `J(-1)` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: fail_error_in_identifying_require_facet_type.carbon:[[@LINE-7]]:30: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:
impl {} as J(-1) {}

// --- fail_error_in_identifying_extend_require_facet_type.carbon
library "[[@TEST_NAME]]";

// Identification tries to form the specific interface `K(array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T: type) {}
constraint J(N: i32) {
  extend require impls K(array(i32, N));
}

// CHECK:STDERR: fail_error_in_identifying_extend_require_facet_type.carbon:[[@LINE+3]]:1-18: error: facet type `J(-1)` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: fail_error_in_identifying_extend_require_facet_type.carbon:[[@LINE-4]]:24-39: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:
impl {} as J(-1) {}

// --- fail_monomorphization_during_period_self_substitution.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface CanFailToMonomorphize(N: Core.IntLiteral) {
  let Z1: array((), N);
  let Z2: type;
}

constraint S(N: Core.IntLiteral) {
  extend require impls CanFailToMonomorphize(N);
}

// When we identify `T as (S(N) where .Z2 impls Y)`
// - we replace `.Self of type S(N)` with `T`
// - that identifies `T as S(N)`
// If the N is invalid, the identify fails in the middle of SubstPeriodSelf
//
// The `S(N)` named constraint indirection ensures that the identify process
// gets to the `.Z2` and attempts `SubstPeriodSelf()` before failing. Without
// the named constraint indirection, we'd just fail to build a specific for
// `CanFailToMonomorphize(N)` and fail without attempting `SubstPeriodSelf()`.
fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z2 impls Y) {}

fn G(generic T: Y) {
  // CHECK:STDERR: fail_monomorphization_during_period_self_substitution.carbon:[[@LINE+5]]:3-10: error: facet type `S(-1) where .(CanFailToMonomorphize(-1).Z2) impls Y` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR: fail_monomorphization_during_period_self_substitution.carbon:[[@LINE-16]]:24-47: note: `CanFailToMonomorphize(N)` evaluates to incomplete type `CanFailToMonomorphize(-1)` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR: fail_monomorphization_during_period_self_substitution.carbon:[[@LINE-22]]:21: note: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR: fail_monomorphization_during_period_self_substitution.carbon:[[@LINE-6]]:1-69: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F(-1, T);
};

// --- fail_monomorphization_identify_in_call.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let Z1: type;
}

constraint S(N: Core.IntLiteral) {
  extend require impls Z where .Z1 = array((), N);
}

fn F(generic N: Core.IntLiteral, generic _: S(N)) {}

fn G(generic T: Z) {
  // CHECK:STDERR: fail_monomorphization_identify_in_call.carbon:[[@LINE+4]]:3-10: error: facet type `S(-1)` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR: fail_monomorphization_identify_in_call.carbon:[[@LINE-7]]:24-49: note: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR: fail_monomorphization_identify_in_call.carbon:[[@LINE-5]]:1-51: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F(-1, T);
};

// --- fail_monomorphization_during_deduction_in_call.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let Z1: type;
}

constraint S(N: Core.IntLiteral) {
  extend require impls Z;
}

fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z1 = array((), N)) {}

fn G(generic T: Z) {
  // CHECK:STDERR: fail_monomorphization_during_deduction_in_call.carbon:[[@LINE+4]]:3-10: error: constructed invalid specific for `S(N) where .(Z.Z1) = array((), N)` from argument [SubstitutingGenericParamType]
  // CHECK:STDERR: fail_monomorphization_during_deduction_in_call.carbon:[[@LINE-4]]:42-73: note: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR: fail_monomorphization_during_deduction_in_call.carbon:[[@LINE-5]]:1-76: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F(-1, T);
};
