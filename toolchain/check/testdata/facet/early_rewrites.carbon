// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/early_rewrites.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/early_rewrites.carbon

// Tests where `ImplWitnessAccess` on the RHS of a rewrite constraint is used in
// other constraints within the same facet type in ways that require the RHS to
// be evaluated to a concrete value before the facet type is resolved (such as
// by passing it as a parameter to a generic class). This can be done either by
// eagerly reading rewrite constraints from the same facet type, or by relying
// on implied constraints which are checked later.

// --- fail_todo_implied_constraint_on_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

class C(T: Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == ()
  // CHECK:STDERR:        ╭─┤ fail_todo_implied_constraint_on_self.carbon:[[@LINE+14]]:25
  // CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
  // CHECK:STDERR:        ·         ─────────┬─────────
  // CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: Z where .Z2 = C(.Self);
  // CHECK:STDERR:        ·                         ┳━━━━━━━
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot convert type `.Self` that implements
  // CHECK:STDERR:        ·                            ┃        `Z` into type implementing
  // CHECK:STDERR:        ·                            ┃        `Z where .(Z.Z1) = ()`
  // CHECK:STDERR:        ·                            ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: Z where .Z2 = C(.Self);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR:        ╭─┤ fail_todo_implied_constraint_on_self.carbon:[[@LINE+15]]:38
// CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
// CHECK:STDERR:        ·         ─────────┬─────────
// CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                     │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing
// CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_implied_constraint_on_associated_constant.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Tuple {}
impl () as Tuple {}

class C(T: Tuple) {}

interface J {
  // Implied: .Z1 impls Tuple
  // CHECK:STDERR:        ╭─┤ fail_todo_implied_constraint_on_associated_constant.carbon:[[@LINE+13]]:25
  // CHECK:STDERR:        │ class C(T: Tuple) {}
  // CHECK:STDERR:        ·         ────┬───
  // CHECK:STDERR:        ·             ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: Z where .Z2 = C(.Z1);
  // CHECK:STDERR:        ·                         ┳━━━━━
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot convert type `.(Z.Z1)` into type
  // CHECK:STDERR:        ·                            ┃        implementing `Tuple`
  // CHECK:STDERR:        ·                            ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: Z where .Z2 = C(.Z1);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_period_self_used_in_access_in_nested_specific.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {}
impl forall [T: Z] T as Y {}

class C(T: Z & Y, U: T.(Z.Z1)) {}

class D;
// This forms the specific `C(.Self)` where `.Self` is of type `Z`
// but we need type `Z & Y` so a conversion takes place. This forms
// a FacetValue with two witnesses. The witness for `Y` is symbolic
// (due to .Self) and points to the impl above. But the witness for
// `Z` is not symbolic and points to the impl being declared here.
// The use of `T.(Z.Z1)` in the class decl ensures we find the value
// from this impl's witness table.
//
// TODO: Right now the type of `U` ends up with an ImplSelfWitness
// which does not resolve the access. So () can't convert to it.
// CHECK:STDERR:        ╭─┤ fail_todo_period_self_used_in_access_in_nested_specific.carbon:[[@LINE+17]]:38
// CHECK:STDERR:        │ class C(T: Z & Y, U: T.(Z.Z1)) {}
// CHECK:STDERR:        · ────────────────┬───────────────
// CHECK:STDERR:        ·                 ╰──┤ while deducing parameters of generic declared here
// CHECK:STDERR:        ·                    │ [DeductionGenericHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self, .Z1) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot implicitly convert
// CHECK:STDERR:        ·                                      │  ┃        expression of type `type` to
// CHECK:STDERR:        ·                                      │  ┃        `.(Z.Z1)`
// CHECK:STDERR:        ·                                      │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                                      ╰────┤ type `type` does not implement
// CHECK:STDERR:        ·                                           │ interface `Core.ImplicitAs(.(Z.Z1))`
// CHECK:STDERR:        ·                                           │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self, .Z1) {}

// --- fail_todo_period_self_used_in_access_in_nested_specific_before_rewrite_of_access.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {}
impl forall [T: Z] T as Y {}

class C(T: Z & Y, U: T.(Z.Z1)) {}

class D;
// This forms the specific `C(.Self)` where `.Self` is of type `Z`
// but we need type `Z & Y` so a conversion takes place. This forms
// a FacetValue with two witnesses. The witness for `Y` is symbolic
// (due to .Self) and points to the impl above. But the witness for
// `Z` is not symbolic and points to the impl being declared here.
// The use of `T.(Z.Z1)` in the class decl ensures we find the value
// from this impl's witness table.
//
// TODO: Right now the type of `U` ends up with an ImplSelfWitness
// which does not resolve the access. So `.Z1` can't convert to it.
// CHECK:STDERR:        ╭─┤ fail_todo_period_self_used_in_access_in_nested_specific_before_rewrite_of_access.carbon:[[@LINE+16]]:25
// CHECK:STDERR:        │ class C(T: Z & Y, U: T.(Z.Z1)) {}
// CHECK:STDERR:        · ────────────────┬───────────────
// CHECK:STDERR:        ·                 ╰──┤ while deducing parameters of generic declared here
// CHECK:STDERR:        ·                    │ [DeductionGenericHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z2 = C(.Self, .Z1) and .Z1 = () {}
// CHECK:STDERR:        ·                         ┳━━━━━━━━━━━━
// CHECK:STDERR:        ·                         ┗━━┫ error: cannot implicitly convert expression of type
// CHECK:STDERR:        ·                         │  ┃        `type` to `.(Z.Z1)`
// CHECK:STDERR:        ·                         │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                         ╰────┤ type `type` does not implement interface
// CHECK:STDERR:        ·                              │ `Core.ImplicitAs(.(Z.Z1))`
// CHECK:STDERR:        ·                              │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z2 = C(.Self, .Z1) and .Z1 = () {}

// --- fail_todo_nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

class C(T: Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR:        ╭─┤ fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:42
  // CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
  // CHECK:STDERR:        ·         ─────────┬─────────
  // CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: (Z where .Z1 = ()) where .Z2 = C(.Self);
  // CHECK:STDERR:        ·                                          ┳━━━━━━━
  // CHECK:STDERR:        ·                                          ┗━━┫ error: cannot convert type `.Self`
  // CHECK:STDERR:        ·                                             ┃        that implements `Z` into
  // CHECK:STDERR:        ·                                             ┃        type implementing
  // CHECK:STDERR:        ·                                             ┃        `Z where .(Z.Z1) = ()`
  // CHECK:STDERR:        ·                                             ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: (Z where .Z1 = ()) where .Z2 = C(.Self);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR:        ╭─┤ fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:38
// CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
// CHECK:STDERR:        ·         ─────────┬─────────
// CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                     │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing
// CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

class C(T: Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR:        ╭─┤ fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+14]]:26
  // CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
  // CHECK:STDERR:        ·         ─────────┬─────────
  // CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: (Z where .Z2 = C(.Self)) where .Z1 = ();
  // CHECK:STDERR:        ·                          ┳━━━━━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot convert type `.Self` that implements
  // CHECK:STDERR:        ·                             ┃        `Z` into type implementing
  // CHECK:STDERR:        ·                             ┃        `Z where .(Z.Z1) = ()`
  // CHECK:STDERR:        ·                             ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: (Z where .Z2 = C(.Self)) where .Z1 = ();
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR:        ╭─┤ fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:38
// CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
// CHECK:STDERR:        ·         ─────────┬─────────
// CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                     │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing
// CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_earlier_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

class C(T: Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR:        ╭─┤ fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:38
  // CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
  // CHECK:STDERR:        ·         ─────────┬─────────
  // CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: Z where .Z1 = () and .Z2 = C(.Self);
  // CHECK:STDERR:        ·                                      ┳━━━━━━━
  // CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
  // CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
  // CHECK:STDERR:        ·                                         ┃        implementing
  // CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
  // CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: Z where .Z1 = () and .Z2 = C(.Self);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR:        ╭─┤ fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:38
// CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
// CHECK:STDERR:        ·         ─────────┬─────────
// CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                     │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing
// CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_later_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

class C(T: Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR:        ╭─┤ fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE+14]]:25
  // CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
  // CHECK:STDERR:        ·         ─────────┬─────────
  // CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                     │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let J1: Z where .Z2 = C(.Self) and .Z1 = ();
  // CHECK:STDERR:        ·                         ┳━━━━━━━
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot convert type `.Self` that implements
  // CHECK:STDERR:        ·                            ┃        `Z` into type implementing
  // CHECK:STDERR:        ·                            ┃        `Z where .(Z.Z1) = ()`
  // CHECK:STDERR:        ·                            ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let J1: Z where .Z2 = C(.Self) and .Z1 = ();
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR:        ╭─┤ fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE+15]]:38
// CHECK:STDERR:        │ class C(T: Z where .Z1 = ()) {}
// CHECK:STDERR:        ·         ─────────┬─────────
// CHECK:STDERR:        ·                  ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                     │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing
// CHECK:STDERR:        ·                                         ┃        `Z where .(Z.Z1) = ()`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- early_rewrite_applied.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Tuple {}
impl () as Tuple {}
class C(T: Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1: Z where .Z1 = () and .Z2 = C(.Z1);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- early_rewrite_applied_from_nested_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Tuple {}
impl () as Tuple {}
class C(T: Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1: (Z where .Z1 = ()) where .Z2 = C(.Z1);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- early_rewrite_applied_extra_indirection.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {
  let Y1: Z;
}

interface Tuple {}
impl () as Tuple {}
class C(T: Tuple) {}

interface J {
  // .Z2 depends on an two nested accesses of rewrite constraints.
  let J1: Y & Z where .Y1 = .Self and .Z1 = () and .Z2 = C(.Y1.Z1);
}

fn F(unused generic T: J) {}

class D;
impl D as Z where .Z1 = () and .Z2 = C(()) {}
impl D as Y where .Y1 = D {}
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- early_rewrite_of_compound_access.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {}
class NeedY(T: Y) {}

impl () as Y {}

// These are written differently but should behave the same.

// Designator access of `.Z1`.
fn F(generic _: Z where .Z1 = () and .Z2 = NeedY(.Z1)) {}

// Compound member access of `.Z1`.
fn G(generic _: Z where .Z1 = () and .Z2 = NeedY(.Self.(Z.Z1))) {}

// --- fail_todo_resolved_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1: type;
}

interface Z {
  let Z1: Y;
  let Z2: type;
  let Z3: type;
}

interface Tuple {}
impl () as Tuple {}
class C(T: Tuple) {}

interface J {
  // Implied: .Y1 impls Tuple, but this is also explicitly required for J1.
  let J1: Z & Y where .Y1 = () and .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
}

fn F(unused generic T: J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Y where .Y1 = () {}
// CHECK:STDERR:        ╭─┤ fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+8]]:25
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = .Self and .Z2 = () and .Z3 = C(()) {}
// CHECK:STDERR:        ·                         ┳━━━━
// CHECK:STDERR:        ·                         ┗━━┫ error: cannot convert type `.Self` that implements
// CHECK:STDERR:        ·                            ┃        `Z` into type implementing `Y`
// CHECK:STDERR:        ·                            ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = .Self and .Z2 = () and .Z3 = C(()) {}
// CHECK:STDERR:        ╭─┤ fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+9]]:31
// CHECK:STDERR: ->     │ fn G(generic T: J where .J1 = D) {
// CHECK:STDERR:        ·                               ┳
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot convert type `D` into type implementing
// CHECK:STDERR:        · ┃        `Y & Z where .(Y.Y1) = () and .(Z.Z1) = .Self as Y and .(Z.Z2) = () and .(Z.Z3) = C(() as Tuple)`
// CHECK:STDERR:        · ┃        [ConversionFailureTypeToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G(generic T: J where .J1 = D) {
  F(T);
}

// --- fail_todo_early_rewrite_from_previous_impls_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y {
  let Y1: type;
}

interface Tuple {}
impl () as Tuple {}
class C(U: Tuple);

constraint NY(V: type) {
  extend require impls Y where .Y1 = V;
}

// CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_from_previous_impls_constraint.carbon:[[@LINE+13]]:59
// CHECK:STDERR:        │ class C(U: Tuple);
// CHECK:STDERR:        ·         ────┬───
// CHECK:STDERR:        ·             ╰──┤ initializing generic parameter `U` declared here
// CHECK:STDERR:        ·                │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F(unused generic T: Z where .Z1 impls NY(()) and .Z2 = C(.Z1.(Y.Y1))) {}
// CHECK:STDERR:        ·                                                           ━━━━━━┳━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot convert type `.(Z.Z1).(Y.Y1)` into type implementing `Tuple`
// CHECK:STDERR:        · ┃        [ConversionFailureTypeToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(unused generic T: Z where .Z1 impls NY(()) and .Z2 = C(.Z1.(Y.Y1))) {}

// --- fail_early_rewrite_correct_interface_wrong_self_access_rewrite_in_period_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y {
  let Y1: type;
}

interface Tuple {}
impl () as Tuple {}
class C(U: Tuple);

constraint NY(V: type) {
  extend require impls Y where .Y1 = V;
}

// This should fail: `T.(Y.Y1)` is rewritten to be `()` but `T.(Z.Z1).(Y.Y1)`
// is used in the argument to `C`.
//
// CHECK:STDERR:        ╭─┤ fail_early_rewrite_correct_interface_wrong_self_access_rewrite_in_period_self.carbon:[[@LINE+13]]:77
// CHECK:STDERR:        │ class C(U: Tuple);
// CHECK:STDERR:        ·         ────┬───
// CHECK:STDERR:        ·             ╰──┤ initializing generic parameter `U` declared here
// CHECK:STDERR:        ·                │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ ... T: Z where .Self impls NY(()) and .Z1 impls Y and .Z2 = C(.Z1.(Y.Y1))) {}
// CHECK:STDERR:        ·                                                             ━━━━━━┳━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot convert type `.(Z.Z1).(Y.Y1)` into type implementing `Tuple`
// CHECK:STDERR:        · ┃        [ConversionFailureTypeToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(unused generic T: Z where .Self impls NY(()) and .Z1 impls Y and .Z2 = C(.Z1.(Y.Y1))) {}

// --- fail_early_rewrite_correct_interface_wrong_self_access_rewrite_in_associated_constant.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y {
  let Y1: type;
}

interface Tuple {}
impl () as Tuple {}
class C(U: Tuple);

constraint NY(V: type) {
  extend require impls Y where .Y1 = V;
}

// This should fail: `T.(Z.Z1).(Y.Y1)` is rewritten to be `()` but `T.(Y.Y1)` is used
// in the argument to `C`.
//
// CHECK:STDERR:        ╭─┤ fail_early_rewrite_correct_interface_wrong_self_access_rewrite_in_associated_constant.carbon:[[@LINE+10]]:79
// CHECK:STDERR: ->     │ ...T: Z where .Z1 impls NY(()) and .Self impls Y and .Z2 = C(Self.(Y.Y1))) {}
// CHECK:STDERR:        ·                                                              ┳━━━
// CHECK:STDERR:        ·                                                              ┗━━┫ error: name
// CHECK:STDERR:        ·                                                                 ┃        `Self`
// CHECK:STDERR:        ·                                                                 ┃        not
// CHECK:STDERR:        ·                                                                 ┃        found
// CHECK:STDERR:        ·                                                                 ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G(unused generic T: Z where .Z1 impls NY(()) and .Self impls Y and .Z2 = C(Self.(Y.Y1))) {}

// --- access_of_rewrite_from_nested_facet_type.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X: type;
  let Y: type;
}

constraint NZ(PeriodY: type) {
  extend require impls Z where .X = PeriodY;
}

fn F(generic T: Z where .Y = {} and .X = NZ(.Y), generic U: T.X) -> U.(Z.X) {
  return {};
}
