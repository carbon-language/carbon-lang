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
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == ()
  // CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE+7]]:26: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z2 = C(.Self);
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z2 = C(.Self);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE+7]]:38: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:                                      ^~~~~~~~
// CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE-21]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_implied_constraint_on_associated_constant.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

interface Tuple {}
impl () as Tuple {}

class C(T:! Tuple) {}

interface J {
  // Implied: .Z1 impls Tuple
  // CHECK:STDERR: fail_todo_implied_constraint_on_associated_constant.carbon:[[@LINE+7]]:26: error: cannot convert type `.(Z.Z1)` into type implementing `Tuple` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z2 = C(.Z1);
  // CHECK:STDERR:                          ^~~~~~
  // CHECK:STDERR: fail_todo_implied_constraint_on_associated_constant.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Tuple) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z2 = C(.Z1);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:43: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! (Z where .Z1 = ()) where .Z2 = C(.Self);
  // CHECK:STDERR:                                           ^~~~~~~~
  // CHECK:STDERR: fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! (Z where .Z1 = ()) where .Z2 = C(.Self);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR: fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:38: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:                                      ^~~~~~~~
// CHECK:STDERR: fail_todo_nested_constraint_visible_in_type_parameter.carbon:[[@LINE-21]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:27: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! (Z where .Z2 = C(.Self)) where .Z1 = ();
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR: fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! (Z where .Z2 = C(.Self)) where .Z1 = ();
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR: fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:38: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:                                      ^~~~~~~~
// CHECK:STDERR: fail_todo_outer_nested_constraint_visible_in_type_parameter.carbon:[[@LINE-21]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_earlier_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:39: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z1 = () and .Z2 = C(.Self);
  // CHECK:STDERR:                                       ^~~~~~~~
  // CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z1 = () and .Z2 = C(.Self);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:38: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:                                      ^~~~~~~~
// CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE-21]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_later_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:26: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z2 = C(.Self) and .Z1 = ();
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z2 = C(.Self) and .Z1 = ();
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
// CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:38: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
// CHECK:STDERR:                                      ^~~~~~~~
// CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE-21]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- early_rewrite_applied.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! Z where .Z1 = () and .Z2 = C(.Z1);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- early_rewrite_applied_from_nested_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! (Z where .Z1 = ()) where .Z2 = C(.Z1);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_early_rewrite_applied_extra_indirection.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

interface Y {
  let Y1:! Z;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! Y & Z where .Y1 = .Self and .Z1 = () and .Z2 = C(.Y1.Z1);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
impl D as Y where .Y1 = D {}
// CHECK:STDERR: fail_todo_early_rewrite_applied_extra_indirection.carbon:[[@LINE+4]]:24: error: cannot convert type `D` into type implementing `Z & Y where .(Y.Y1) = .Self as Z and .(Z.Z1) = () and .(Z.Z2) = C(() as Tuple)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn G(T:! J where .J1 = D) {
// CHECK:STDERR:                        ^
// CHECK:STDERR:
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_resolved_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

interface Z {
  let Z1:! Y;
  let Z2:! type;
  let Z3:! type;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

interface J {
  // Implied: .Y1 impls Tuple, but this is also explicitly required for J1.
  let J1:! Z & Y where .Y1 = () and .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Y where .Y1 = () {}
// CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+4]]:25: error: cannot convert type `.Self` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
// CHECK:STDERR: impl D as Z where .Z1 = .Self and .Z2 = () and .Z3 = C(()) {}
// CHECK:STDERR:                         ^~~~~
// CHECK:STDERR:
impl D as Z where .Z1 = .Self and .Z2 = () and .Z3 = C(()) {}
// CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+4]]:24: error: cannot convert type `D` into type implementing `Y & Z where .(Y.Y1) = () and .(Z.Z1) = .Self as Y and .(Z.Z2) = () and .(Z.Z3) = C(() as Tuple)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn G(T:! J where .J1 = D) {
// CHECK:STDERR:                        ^
// CHECK:STDERR:
fn G(T:! J where .J1 = D) {
  F(T);
}
