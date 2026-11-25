// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon

// --- common.carbon
library "[[@TEST_NAME]]";

class C {};
class D {};

impl C as Core.AddWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C { return {}; }
}
impl C as Core.AddAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

class C2 {};

impl C2 as Core.Add {
  fn Op(unused self: C2, unused other: C2) -> C2 { return {}; }
}
impl C2 as Core.AddAssign {
  fn Op(unused ref self: C2, unused other: C2) {}
}

class E {};

impl E as Core.AddWith(E) where .Result = C {
  fn Op(unused self: E, unused other: E) -> C { return {}; }
}

fn GenericAdd[T: Core.Add](a: T, b: T) -> T {
  return a + b;
}

fn GenericAddAssign[T: Core.AddAssign](ref a: T, b: T) {
  a += b;
}

// --- fail_add_with.carbon
library "[[@TEST_NAME]]";

import library "common";

fn Test(a: C, b: D, z: bool) -> C {
  if (z) {
    return a + a;
  } else {
    // CHECK:STDERR:        ╭─┤ fail_add_with.carbon:[[@LINE+8]]:12
    // CHECK:STDERR: ->     │     return a + b;
    // CHECK:STDERR:        ·            ┳━━━━
    // CHECK:STDERR:        ·            ┗━━┫ error: cannot access member of interface `Core.AddWith(D)` in
    // CHECK:STDERR:        ·               ┃        type `C` that does not implement that interface
    // CHECK:STDERR:        ·               ┃        [MissingImplInMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    return a + b;
  }
}

// --- fail_add_assign_with.carbon
library "[[@TEST_NAME]]";

import library "common";

fn TestAssign(b: D) {
  var a: C = {};
  a += a;
  // CHECK:STDERR:        ╭─┤ fail_add_assign_with.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   a += b;
  // CHECK:STDERR:        ·   ┳━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `Core.AddAssignWith(D)` in type
  // CHECK:STDERR:        ·      ┃        `C` that does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  a += b;
}

// --- fail_impl_add_wrong_result.carbon
library "[[@TEST_NAME]]";

import library "common";

class F {};

// CHECK:STDERR:        ╭─┤ fail_impl_add_wrong_result.carbon:[[@LINE+8]]:11
// CHECK:STDERR: ->     │ impl F as Core.Add where .Result = C {
// CHECK:STDERR:        ·           ┳━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: associated constant Result given value `C` that is
// CHECK:STDERR:        ·              ┃        different from previously given value `F`
// CHECK:STDERR:        ·              ┃        [AssociatedConstantGivenDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl F as Core.Add where .Result = C {
  fn Op(unused self: F, unused other: F) -> C { return {}; }
}

// --- fail_add_named_constraint.carbon
library "[[@TEST_NAME]]";

import library "common";

fn TestNamedConstraint(a: C2, b: D, z: bool) -> C2 {
  if (z) {
    return a + a;
  } else {
    // CHECK:STDERR:        ╭─┤ fail_add_named_constraint.carbon:[[@LINE+8]]:12
    // CHECK:STDERR: ->     │     return a + b;
    // CHECK:STDERR:        ·            ┳━━━━
    // CHECK:STDERR:        ·            ┗━━┫ error: cannot access member of interface `Core.AddWith(D)` in
    // CHECK:STDERR:        ·               ┃        type `C2` that does not implement that interface
    // CHECK:STDERR:        ·               ┃        [MissingImplInMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    return a + b;
  }
}

// --- fail_add_assign_named_constraint.carbon
library "[[@TEST_NAME]]";

import library "common";

fn TestNamedConstraintAssign(b: D) {
  var a: C2 = {};
  a += a;
  // CHECK:STDERR:        ╭─┤ fail_add_assign_named_constraint.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   a += b;
  // CHECK:STDERR:        ·   ┳━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `Core.AddAssignWith(D)` in type
  // CHECK:STDERR:        ·      ┃        `C2` that does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  a += b;
}

// --- fail_generic_add_no_impl.carbon
library "[[@TEST_NAME]]";

import library "common";

fn Test(d: D) {
  // CHECK:STDERR:        ╭─┤ fail_generic_add_no_impl.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   GenericAdd(d, d);
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: cannot convert type `D` into type implementing `Core.Add`
  // CHECK:STDERR:        ·              ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ common.carbon
  // CHECK:STDERR:        │ fn GenericAdd[T: Core.Add](a: T, b: T) -> T {
  // CHECK:STDERR:        · ──────────────────────┬──────────────────────
  // CHECK:STDERR:        ·                       ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·                          │ [DeductionGenericHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  GenericAdd(d, d);
}

// --- fail_generic_add_wrong_result.carbon
library "[[@TEST_NAME]]";

import library "common";

fn Test(e: E) {
  // CHECK:STDERR:        ╭─┤ fail_generic_add_wrong_result.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   GenericAdd(e, e);
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: cannot convert type `E` into type implementing `Core.Add`
  // CHECK:STDERR:        ·              ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ common.carbon
  // CHECK:STDERR:        │ fn GenericAdd[T: Core.Add](a: T, b: T) -> T {
  // CHECK:STDERR:        · ──────────────────────┬──────────────────────
  // CHECK:STDERR:        ·                       ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·                          │ [DeductionGenericHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  GenericAdd(e, e);
}

// --- fail_generic_add_assign_no_impl.carbon
library "[[@TEST_NAME]]";

import library "common";

fn Test(d: D) {
  var dv: D = {};
  // CHECK:STDERR:        ╭─┤ fail_generic_add_assign_no_impl.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   GenericAddAssign(ref dv, d);
  // CHECK:STDERR:        ·   ━━━┳━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: cannot convert type `D` into type implementing `Core.AddAssign`
  // CHECK:STDERR:        ·         ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ common.carbon
  // CHECK:STDERR:        │ fn GenericAddAssign[T: Core.AddAssign](ref a: T, b: T) {
  // CHECK:STDERR:        · ─────────────────────────┬──────────────────────────────
  // CHECK:STDERR:        ·                          ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·                             │ [DeductionGenericHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  GenericAddAssign(ref dv, d);
}
