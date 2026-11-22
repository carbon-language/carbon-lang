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

// Unlike `array(T, N)`, this fails eagerly on an invalid bound.
fn Array(T: type, N: Core.IntLiteral) -> type = "array.make_type";

// Identification tries to form the specific interface `K(Array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T: type) {}
constraint L(N: i32) {
  // CHECK:STDERR:        ╭─┤ fail_error_in_identifying_require_facet_type.carbon:[[@LINE+14]]:30
  // CHECK:STDERR: ->     │   require impls K(Array(i32, N));
  // CHECK:STDERR:        ·   ───────────────┬───────────┳───
  // CHECK:STDERR:        ·                  │           ┗━━┫ error: array bound of -1 is negative
  // CHECK:STDERR:        ·                  │              ┃        [ArrayBoundNegative]
  // CHECK:STDERR:        ·                  ╰────────────────┤ in monomorphization of `require` here
  // CHECK:STDERR:        ·                                   │ [ResolvingSpecificHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ impl {} as J(-1) {}
  // CHECK:STDERR:        ·            ──┬──
  // CHECK:STDERR:        ·              ╰──┤ in identification of facet type `J(-1)` here
  // CHECK:STDERR:        ·                 │ [InFacetTypeIdentification]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require impls K(Array(i32, N));
}
constraint J(N: i32) {
  require impls L(N);
}

impl {} as J(-1) {}

// --- fail_error_in_identifying_extend_require_facet_type.carbon
library "[[@TEST_NAME]]";

// Unlike `array(T, N)`, this fails eagerly on an invalid bound.
fn Array(T: type, N: Core.IntLiteral) -> type = "array.make_type";

// Identification tries to form the specific interface `K(Array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T: type) {}
constraint J(N: i32) {
  // CHECK:STDERR:        ╭─┤ fail_error_in_identifying_extend_require_facet_type.carbon:[[@LINE+13]]:24
  // CHECK:STDERR: ->     │   extend require impls K(Array(i32, N));
  // CHECK:STDERR:        ·                        ━━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·                                ┗━━┫ error: array bound of -1 is negative
  // CHECK:STDERR:        ·                                   ┃        [ArrayBoundNegative]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ impl {} as J(-1) {}
  // CHECK:STDERR:        ·            ──┬──
  // CHECK:STDERR:        ·              ╰──┤ in monomorphization of `J(-1)` here [ResolvingSpecificHere]
  // CHECK:STDERR:        ·              ╰────┤ in identification of facet type `J(-1)` here
  // CHECK:STDERR:        ·                   │ [InFacetTypeIdentification]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend require impls K(Array(i32, N));
}

impl {} as J(-1) {}

// --- fail_monomorphization_during_period_self_substitution.carbon
library "[[@TEST_NAME]]";

// Unlike `array(T, N)`, this fails eagerly on an invalid bound.
fn Array(T: type, N: Core.IntLiteral) -> type = "array.make_type";

interface Y {}
interface CanFailToMonomorphize(N: Core.IntLiteral) {
  // CHECK:STDERR:        ╭─┤ fail_monomorphization_during_period_self_substitution.carbon:[[@LINE+30]]:21
  // CHECK:STDERR: ->     │   let Z1: Array((), N);
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: array bound of -1 is negative
  // CHECK:STDERR:        ·                        ┃        [ArrayBoundNegative]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │   extend require impls CanFailToMonomorphize(N);
  // CHECK:STDERR:        ·                        ┬───────────────────────
  // CHECK:STDERR:        ·                        ╰──┤ in monomorphization of `CanFailToMonomorphize(-1)`
  // CHECK:STDERR:        ·                        │  │ here
  // CHECK:STDERR:        ·                        │  │ [ResolvingSpecificHere]
  // CHECK:STDERR:        ·                        ╰────┤ in completion of type `CanFailToMonomorphize(-1)`
  // CHECK:STDERR:        ·                             │ here
  // CHECK:STDERR:        ·                             │ [InTypeCompletion]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z2 impls Y) {}
  // CHECK:STDERR:        ·    ┬                                     ┬────────────────────────
  // CHECK:STDERR:        ·    ╰────────────────────────────────────────┤ while deducing parameters of
  // CHECK:STDERR:        ·                                          │  │ generic declared here
  // CHECK:STDERR:        ·                                          │  │ [DeductionGenericHere]
  // CHECK:STDERR:        ·                                          ╰────┤ in monomorphization of `S(-1)`
  // CHECK:STDERR:        ·                                               │ here
  // CHECK:STDERR:        ·                                               │ [ResolvingSpecificHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │   F(-1, T);
  // CHECK:STDERR:        ·   ────┬───
  // CHECK:STDERR:        ·       ╰──┤ while substituting generic arguments into the type of `_`
  // CHECK:STDERR:        ·          │ [SubstitutingGenericParamType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let Z1: Array((), N);
  let Z2: type;
}

constraint S(N: Core.IntLiteral) {
  extend require impls CanFailToMonomorphize(N);
}

// When we call `F` and convert `T` to `S(N) where .Z2 impls Y`:
// - we identify `T as (S(N) where .Z2 impls Y)` for impl lookup
// - that replaces `.Self of type S(N)` with `T`
// - that identifies `T as S(N)`
// If the N is invalid, the identify fails in the middle of SubstPeriodSelf
//
// The `S(N)` named constraint indirection ensures that the identify process
// gets to the `.Z2` and attempts `SubstPeriodSelf()` before failing. Without
// the named constraint indirection, we'd just fail to build a specific for
// `CanFailToMonomorphize(N)` and fail without attempting `SubstPeriodSelf()`.
fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z2 impls Y) {}

fn G(generic T: Y) {
  F(-1, T);
};

// --- fail_monomorphization_identify_in_call.carbon
library "[[@TEST_NAME]]";

// Unlike `array(T, N)`, this fails eagerly on an invalid bound.
fn Array(T: type, N: Core.IntLiteral) -> type = "array.make_type";

interface Y {}
interface Z {
  let Z1: type;
}

constraint S(N: Core.IntLiteral) {
  // CHECK:STDERR:        ╭─┤ fail_monomorphization_identify_in_call.carbon:[[@LINE+18]]:24
  // CHECK:STDERR: ->     │   extend require impls Z where .Z1 = Array((), N);
  // CHECK:STDERR:        ·                        ━━━━━━━━━━━━━┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                     ┗━━┫ error: array bound of -1 is negative
  // CHECK:STDERR:        ·                                        ┃        [ArrayBoundNegative]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn F(generic N: Core.IntLiteral, generic _: S(N)) {}
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │   F(-1, T);
  // CHECK:STDERR:        ·   ────┬───
  // CHECK:STDERR:        ·       ╰──┤ in monomorphization of `S(-1)` here [ResolvingSpecificHere]
  // CHECK:STDERR:        ·       ╰────┤ in identification of facet type `S(-1)` here
  // CHECK:STDERR:        ·            │ [InFacetTypeIdentification]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend require impls Z where .Z1 = Array((), N);
}

fn F(generic N: Core.IntLiteral, generic _: S(N)) {}

fn G(generic T: Z) {
  F(-1, T);
};

// --- fail_monomorphization_during_deduction_in_call.carbon
library "[[@TEST_NAME]]";

// Unlike `array(T, N)`, this fails eagerly on an invalid bound.
fn Array(T: type, N: Core.IntLiteral) -> type = "array.make_type";

interface Y {}
interface Z {
  let Z1: type;
}

constraint S(N: Core.IntLiteral) {
  extend require impls Z;
}

// CHECK:STDERR:        ╭─┤ fail_monomorphization_during_deduction_in_call.carbon:[[@LINE+16]]:42
// CHECK:STDERR: ->     │ fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z1 = Array((), N)) {}
// CHECK:STDERR:        ·    ┬                                     ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·    │                                     ┗━━┫ error: array bound of -1 is
// CHECK:STDERR:        ·    │                                        ┃        negative
// CHECK:STDERR:        ·    │                                        ┃        [ArrayBoundNegative]
// CHECK:STDERR:        ·    ╰──────────────────────────────────────────┤ while deducing parameters of
// CHECK:STDERR:        ·                                               │ generic declared here
// CHECK:STDERR:        ·                                               │ [DeductionGenericHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        │   F(-1, T);
// CHECK:STDERR:        ·   ────┬───
// CHECK:STDERR:        ·       ╰──┤ while substituting generic arguments into the type of `_`
// CHECK:STDERR:        ·          │ [SubstitutingGenericParamType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(generic N: Core.IntLiteral, generic _: S(N) where .Z1 = Array((), N)) {}

fn G(generic T: Z) {
  F(-1, T);
};
