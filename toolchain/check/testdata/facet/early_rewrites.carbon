// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
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
  let J1:! Z where .Z2 = C(.Self);
}

fn F(T:! J) {}
// Satisfies the implied constraint in J.J1.
fn G(T:! Z where .Z1 = () and .Z2 = C(.Self)) {
  // CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `Z where .(Z.Z1) = () and .(Z.Z2) = C(.Self as Z where .(Z.Z1) = ())` into type implementing `J` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_implied_constraint_on_self.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! J) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
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
  // Implied: .Z1 == ()
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

// --- nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! (Z where .Z1 = ()) where .Z2 = C(.Self);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- outer_nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! (Z where .Z2 = C(.Self)) where .Z1 = ();
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- earlier_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! Z where .Z1 = () and .Z2 = C(.Self);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- later_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  let J1:! Z where .Z2 = C(.Self) and .Z1 = ();
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(.Self) {}
fn G(T:! J where .J1 = D) {
  F(T);
}

// --- fail_todo_early_rewrite_applied.carbon
interface Z {
  let Z1:! type;
  let Z2:! type;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

interface J {
  // Implied: .Z1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_early_rewrite_applied.carbon:[[@LINE+7]]:39: error: cannot convert type `.(Z.Z1)` into type implementing `Tuple` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z1 = () and .Z2 = C(.Z1);
  // CHECK:STDERR:                                       ^~~~~~
  // CHECK:STDERR: fail_todo_early_rewrite_applied.carbon:[[@LINE-7]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Tuple) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z1 = () and .Z2 = C(.Z1);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Z where .Z1 = () and .Z2 = C(()) {}
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
  // Implied: .Y1 == (), but this is also explicitly required for J1.
  // CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+8]]:33: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let J1:! Z & Y where .Y1 = () where .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
  // CHECK:STDERR:                                 ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let J1:! Z & Y where .Y1 = () where .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let J1:! Z & Y where .Y1 = () where .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
}

fn F(T:! J) {}

class D;
// Satisfies the implied constraint in J.J1.
impl D as Y where .Y1 = () {}
impl D as Z where .Z1 = .Self and .Z2 = () and .Z3 = C(()) {}
fn G(T:! J where .J1 = D) {
  F(T);
}
