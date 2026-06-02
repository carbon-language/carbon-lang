// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon

// --- self_impls_modifies_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(unused T:! I where .X = ()) {}

fn G(T:! I where .Self impls (I where .X = ())) {
  F(T);
}

// --- fail_self_impls_modifies_assoc_constant_type_differs.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(unused T:! I where .X = ()) {}

fn G(T:! I where .Self impls (I where .X = {})) {
  // CHECK:STDERR: fail_self_impls_modifies_assoc_constant_type_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = {}` into type implementing `I where .(I.X) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_self_impls_modifies_assoc_constant_type_differs.carbon:[[@LINE-6]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused T:! I where .X = ()) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// --- fail_where_impls_tests_associated_constant_of_generic_type_non_final_impl.carbon
library "[[@TEST_NAME]]";

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M { let M0:! type; }
impl forall [U:! L] C(U) as M where .M0 = {} {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(unused U:! type where C(.Self) impls (M where .M0 = {})) {}

fn G(T:! L) {
  // We have no final impl for `C(T) as M`, so we don't know the value of
  // `(C(T) as M).M0` concretely, so we don't know that it is `{}` and we can't
  // convert assuming it is.
  //
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_non_final_impl.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where C(.Self) impls M and C(.Self).(M.M0) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_non_final_impl.carbon:[[@LINE-10]]:13: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused U:! type where C(.Self) impls (M where .M0 = {})) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// --- where_impls_tests_associated_constant_of_generic_type.carbon
library "[[@TEST_NAME]]";

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M { let M0:! type; }
final impl forall [U:! L] C(U) as M where .M0 = {} {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(unused U:! type where C(.Self) impls (M where .M0 = {})) {}

fn G(T:! L) {
  F(T);
}

// --- fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon
library "[[@TEST_NAME]]";

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M { let M0:! type; }
final impl forall [U:! L] C(U) as M where .M0 = () {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(unused U:! type where C(.Self) impls (M where .M0 = {})) {}

fn G(T:! L) {
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where C(.Self) impls M and C(.Self).(M.M0) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon:[[@LINE-6]]:13: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused U:! type where C(.Self) impls (M where .M0 = {})) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// --- fail_where_impls_tests_associated_constant_of_generic_interface_non_final_impl.carbon
library "[[@TEST_NAME]]";

class C {}

// C impls M(U) if U impls L.
interface L {}
interface M(U:! type) { let M0:! type; }
impl forall [U:! L] C as M(U) where .M0 = {} {}

// U requires that C impls M(.Self).
// - C impls M(.Self) can be rewritten as C impls M(U).
// - C impls M(U) if U impls L => Requires U impls L.
fn F(unused U:! type where C impls (M(.Self) where .M0 = {})) {}

fn G(T:! L) {
  // We have no final impl for `C as M(T)`, so we don't know the value of
  // `(C as M(T)).M0` concretely, so we don't know that it is `{}` and we can't
  // convert assuming it is.
  //
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_interface_non_final_impl.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where C impls M(.Self) and C.(M(.Self).M0) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_interface_non_final_impl.carbon:[[@LINE-10]]:13: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused U:! type where C impls (M(.Self) where .M0 = {})) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// --- where_impls_tests_associated_constant_of_generic_interface.carbon
library "[[@TEST_NAME]]";

class C {}

// C impls M(U) if U impls L.
interface L {}
interface M(U:! type) { let M0:! type; }
final impl forall [U:! L] C as M(U) where .M0 = {} {}

// U requires that C impls M(.Self).
// - C impls M(.Self) can be rewritten as C impls M(U).
// - C impls M(U) if U impls L => Requires U impls L.
fn F(unused U:! type where C impls (M(.Self) where .M0 = {})) {}

fn G(T:! L) {
  F(T);
}

// --- fail_where_impls_tests_associated_constant_of_generic_interface_type_differs.carbon
library "[[@TEST_NAME]]";

class C {}

// C impls M(U) if U impls L.
interface L {}
interface M(U:! type) { let M0:! type; }
final impl forall [U:! L] C as M(U) where .M0 = () {}

// U requires that C impls M(.Self).
// - C impls M(.Self) can be rewritten as C impls M(U).
// - C impls M(U) if U impls L => Requires U impls L.
fn F(unused U:! type where C impls (M(.Self) where .M0 = {})) {}

fn G(T:! L) {
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_interface_type_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where C impls M(.Self) and C.(M(.Self).M0) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_interface_type_differs.carbon:[[@LINE-6]]:13: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused U:! type where C impls (M(.Self) where .M0 = {})) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// --- self_in_interface_generic_param_unconstrained.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface I(T:! type) {}

fn F(unused T:! I(.Self) where .Self impls Z) {}

fn G(T:! Z & I(.Self)) {
  F(T);
}

// --- fail_todo_self_in_interface_generic_param_constrained.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface I(T:! Z) {}

// TODO: I(.Self) introduces an implied constraint `.Self impls Z`, which is
// satisfied and checked at the end of the fn signature.
//
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE+7]]:17: error: cannot convert type `.Self` that implements `type` into type implementing `Z` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn F(unused T:! I(.Self) where .Self impls Z) {}
// CHECK:STDERR:                 ^~~~~~~~
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE-8]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface I(T:! Z) {}
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
fn F(unused T:! I(.Self) where .Self impls Z) {}

// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE+7]]:14: error: cannot convert type `.Self` that implements `type` into type implementing `Z` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn G(T:! Z & I(.Self)) {
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE-17]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface I(T:! Z) {}
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
fn G(T:! Z & I(.Self)) {
  F(T);
}

// --- where_period_self_rhs_sees_lhs.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y {}
interface X(T:! Z & Y) {}

constraint N {
  require impls Y;
}

// The RHS of `where` can see the extend constraints on the LHS of `where`.
fn F(_:! Z & N where .Self impls X(.Self)) {}

fn G() {
  class C;
  impl C as Z {}
  impl C as Y {}
  impl C as X(C) {}
  F(C);
}

// --- where_type_rhs_sees_lhs.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y {}
interface X(T:! Z & Y) {}

constraint N {
  require impls Y;
}

class R(T:! Z & Y);

// The RHS of `where` can see the extend constraints on the LHS of `where`.
fn F(_:! Z & N where R(.Self) impls X(.Self)) {}

fn G() {
  class C;
  impl C as Z {}
  impl C as Y {}
  impl R(C) as X(C) {}
  F(C);
}

// --- fail_todo_where_period_self_rhs_impls_nested_period_self.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y {}
interface X(T:! Z & Y) {}

constraint N {
  require impls Y;
}

// TODO: The inner nested facet type (`N where...`) does not know about
// constraints on the outer facet type (`Z where...`). But it should produce an
// implied constraint that `.Self impls Z & Y` that is checked once the full
// facet type is known.
//
// CHECK:STDERR: fail_todo_where_period_self_rhs_impls_nested_period_self.carbon:[[@LINE+7]]:51: error: cannot convert type `.Self` that implements `N` into type implementing `Z & Y` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn F(_:! Z where .Self impls (N where .Self impls X(.Self))) {}
// CHECK:STDERR:                                                   ^~~~~~~~
// CHECK:STDERR: fail_todo_where_period_self_rhs_impls_nested_period_self.carbon:[[@LINE-14]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface X(T:! Z & Y) {}
// CHECK:STDERR:             ^~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z where .Self impls (N where .Self impls X(.Self))) {}

fn G() {
  class C;
  impl C as Z {}
  impl C as Y {}
  impl C as X(C) {}
  F(C);
}

// --- associated_const_impls_interface_with_period_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

interface J(T:! I) {}

// There is a .Self reference on the LHS and RHS of the `impls` constraint,
// which must be replaced, each with `C as I`
fn F(_:! I where .I1 impls J(.Self)) {}

fn G() {
  class C;
  class D;

  impl C as I where .I1 = D {}
  impl D as J(C) {}

  F(C);
}

// --- fail_concrete_witness_without_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}
interface J {}

fn F(_:! I where .I1 impls J) {}

fn G() {
  class C;

  // This identifies `C as (I where .I1 impls J)`, which replaces `.Self.I1`
  // with `C.(I.I1)`. This is a concrete lookup since C and I are both concrete,
  // but it doesn't find anything as there is no impl.
  //
  // This tests that we correctly handle the case where the witness in an
  // ImplWitnessAccess would become concrete without being able to provide any
  // value, since it's still a `LookupImplWitness` type of witness.

  // CHECK:STDERR: fail_concrete_witness_without_impl.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I where .(I.I1) impls J` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(C);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_concrete_witness_without_impl.carbon:[[@LINE-16]]:6: note: initializing generic parameter `_` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(_:! I where .I1 impls J) {}
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(C);
}

// --- convert_to_period_self_preserves_self_facet_impls.carbon
library "[[@TEST_NAME]]";

interface K {}
interface J {}
interface I {
  let I1:! type;
}

fn F(unused U:! I & J where .I1 impls K) {}

impl forall [T:! type] T as J {}

fn G(T:! I where .I1 impls K) {
  // Replaces `.Self` with `T`, which converts `T` to `I & J`. Doing so has to
  // invent a witness for `J`, and this tests we do so without losing the
  // information that `.I1 impls K`.
  F(T);
}

// --- find_witness_from_facet_in_lookup_target_facet_type.carbon
library "[[@TEST_NAME]]";

class D(T:! type) {}
class E {}
interface X(T:! type) {}
interface V(T:! type) {}

constraint W(T:! type) {
  require Self impls X(T);
}

fn F(B:! W(E), C:! type where E impls V(.Self)) {
  // These find a witness for `E as X(E)` from the facet `B`.
  E as (V(C) where B impls X(.Self));
  E as (V(C) where B impls X(E));
}

// --- find_witness_from_facet_in_specific_in_lookup_target_facet_type.carbon
library "[[@TEST_NAME]]";

class D(T:! type) {}
class E {}
interface X(T:! type) {}
interface V(T:! type) {}

constraint W(T:! type) {
  require D(Self) impls X(T);
}

fn F(B:! W(E), C:! type where E impls V(.Self)) {
  // These find a witness for `E as X(E)` from the facet `B`.
  E as (V(C) where D(B) impls X(.Self));
  E as (V(C) where D(B) impls X(E));
}

// --- facet_has_witness_for_impl_witness_access.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Y {}
interface Z {
  // A type type, which has no witnesses.
  let Z1:! type;
  // A facet type, but not one that provides a witness for `.Z2 as Y`.
  let Z2:! type where .Self impls X;
}
fn F(T:! Z where .Z1 impls Y and .Z2 impls Y) {
  // `T.Z1` is a `type` so this query self is ImplWitnessAccess(Z1).
  T.Z1 as Y;
  // `T.Z2` has a facet type so this query self is FacetAccessType(ImplWitnessAccess(Z2)).
  T.Z2 as Y;
}

// --- impl_witness_access_has_access.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

interface Y {}
interface Z {
  // Note that the facet type is not an exact type match for `Y` so that we
  // cause an actual impl lookup to happen when we convert `.Z1` to `Y`.
  let Z1:! type where .Self impls Y and C(.Self) impls Y;
}
fn F(T:! Z) {
  T.Z1 as Y;
  C(T.Z1) as Y;
}

// --- period_self_in_type_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {}

constraint GivesY {
  require impls Y;
}

class C(T:! type);

fn F(T:! Z where C(.Self) impls GivesY) {
  C(T) as Y;
}

// --- impl_witness_access_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {}

constraint GivesY {
  require impls Y;
}

// This type places a `.Self` (in the .Z1) into the identified facet type as
// `.Self impls Y` through the named constraint. It ensures that the `.Self` is
// correctly replaced by `T` in order to compare equal with `T.Z1` in impl
// lookup.
fn F(T:! Z where .Z1 impls GivesY) {
  T.Z1 as Y;
}
