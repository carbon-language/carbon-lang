// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/validate_rewrite_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/validate_rewrite_constraints.carbon

// --- facet_value.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

fn F(unused generic T: I where .X = ()) {}

fn G(generic T: I where .X = ()) {
  F(T);
}

// --- generic_interface_facet_value.carbon
library "[[@TEST_NAME]]";

interface I(T: type) { let X: type; }

class C;
final impl forall [T: type] T as I(()) where .X = () {}
final impl forall [T: type] T as I({}) where .X = {} {}

fn F(generic U: type, unused generic T: I(U) where .X = ()) {}
fn G() {
  F((), C);
}

// --- fail_generic_interface_facet_value_wrong_specific_impl.carbon
library "[[@TEST_NAME]]";

interface I(T: type) { let X: type; }

class C;
final impl forall [T: type] T as I(()) where .X = () {}
final impl forall [T: type] T as I({}) where .X = {} {}

fn F(generic U: type, unused generic T: I(U) where .X = ()) {}
fn G() {
  // This finds the impl where `.X = {}`, but F requires that `.X = ()`.
  //
  // CHECK:STDERR: fail_generic_interface_facet_value_wrong_specific_impl.carbon:[[@LINE+3]]:3-10: error: cannot convert type `C` into type implementing `I({}) where {} = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_generic_interface_facet_value_wrong_specific_impl.carbon:[[@LINE-5]]:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F({}, C);
}

// --- dependent_rules.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J { let Y: type; }

fn F(generic T: I & J where .X = .Y) {
  // Allowed since `T impls I`, `T impls J`, and has constraint providing
  // T.(I.X) = T.(J.Y)`.
  F(T);
}

// --- fail_convert.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

fn F(unused generic T: I where .X = {.a: (), .b: {}}) {}

fn H() {
  class C;
  impl C as I where .X = {.b: {}, .a: ()} {}
  // CHECK:STDERR: fail_convert.carbon:[[@LINE+3]]:3-6: error: cannot convert type `C` into type implementing `I where .(I.X) = {.a: (), .b: {}}` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_convert.carbon:[[@LINE-6]]:21-52: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(C);
}

fn G(generic T: I where .X = {.b: {}, .a: ()}) {
  // CHECK:STDERR: fail_convert.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I where .(I.X) = {.b: {}, .a: ()}` into type implementing `I where .(I.X) = {.a: (), .b: {}}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_convert.carbon:[[@LINE-13]]:21-52: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- fail_todo_dependent_rules_compound.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J { let Y: type; }

// CHECK:STDERR: fail_todo_dependent_rules_compound.carbon:[[@LINE+4]]:30: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_dependent_rules_compound.carbon:[[@LINE+2]]:30: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR:
fn F(generic T: I & J where .(I.X) = .(J.Y)) {
  // Allowed since `T impls I`, `T impls J`, and has constraint providing
  // T.(I.X) = T.(J.Y)`.
  F(T);
}

// --- parameterized_interface.carbon
library "[[@TEST_NAME]]";

interface I(T: type) { let X: type; }

final impl forall [J: I(()) where .X = ()] J as I({}) where .X = {} {}

fn F(unused generic T: I({}) where .X = {}) {}

fn G(generic T: I(()) where .X = ()) {
  F(T);
}

// --- fail_todo_parameterized_interface_compound.carbon
library "[[@TEST_NAME]]";

interface I(T: type) { let X: type; }

final impl forall [J: I(())] J as I({}) where .X = {} {}

// CHECK:STDERR: fail_todo_parameterized_interface_compound.carbon:[[@LINE+4]]:38: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_parameterized_interface_compound.carbon:[[@LINE+2]]:38: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR:
fn F(generic T: I(()) & I({}) where .(I(()).X) = () and .(I({}).X) = {}) {}

fn G(generic T: I(()) where .X = ()) {
  F(T);
}

// --- fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon
library "[[@TEST_NAME]]";

interface I(T: type) { let X: type; }

final impl forall [J: I(()) where .X = {}] J as I({}) where .X = {} {}

fn F(unused generic T: I({}) where .X = {}) {}

fn G(generic T: I(()) where .X = ()) {
  // CHECK:STDERR: fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I(()) where .(I(()).X) = ()` into type implementing `I({}) where .(I({}).X) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon:[[@LINE-4]]:21-42: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- source_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}

fn F(unused generic T: I where .X1 = () and .X2 = ()) {}

fn G() {
  class C;
  impl C as I where .X1 = .X2 and .X2 = () {}
  F(C);
}

fn H(generic T: I where .X1 = .X2 and .X2 = ()) {
  F(T);
}

// --- target_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}

fn F(unused generic T: I where .X1 = .X2 and .X2 = ()) {}

fn G() {
  class C;
  impl C as I where .X1 = () and .X2 = () {}
  F(C);
}

fn H(generic T: I where .X1 = () and .X2 = ()) {
  F(T);
}

// --- both_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}

fn F(unused generic T: I where .X1 = .X2 and .X2 = ()) {}

fn G() {
  class C1;
  impl C1 as I where .X1 = .X2 and .X2 = () {}
  F(C1);

  class C2;
  impl C2 as I where .X1 = () and .X2 = .X1 {}
  F(C2);
}

fn H1(generic T: I where .X1 = .X2 and .X2 = ()) {
  F(T);
}

fn H2(generic T: I where .X1 = () and .X2 = .X1) {
  F(T);
}

// --- fail_error_in_witness_table.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}

fn F(unused generic T: I where .X1 = .X2) {}

class C;
// .X2 is not set in the impl definition, so its value is an ErrorInst
// in the impl's witness table.
//
// CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE+3]]:1-28: error: associated constant X2 not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE-10]]:7-14: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
impl C as I where .X1 = () {}

fn G() {
  // CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE+3]]:3-6: error: cannot convert type `C` into type implementing `I where .(I.X1) = .(I.X2)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE-13]]:21-40: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(C);
}

// --- impl_provides_rewrite_requirements.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J { let Y: type; }

class C;

final impl forall [T: I] T as J where .Y = C {}

fn F(unused generic T: I & J where .Y = C) {}
fn G(generic T: I) {
  F(T);
}

// --- facet_provides_rewrite_requirements.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J { let Y: type; }

class C;

final impl forall [T: J] T as I {}

fn F(unused generic T: I & J where .Y = C) {}
fn G(generic T: J where .Y = C) {
  F(T);
}

// --- fail_non_final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

impl forall [U: type] U as I where .X = () {}

fn F(unused generic T: I where .X = ()) {}

class C(T: type);

fn G(generic T: type) {
  // The type `C(T)` is generic so that the resulting impl witness will not be
  // effectively final.
  //
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE+3]]:3-9: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE-9]]:21-38: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(C(T));
}

// --- fail_non_final_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

impl forall [U: type] U as I where .X = () {}

class C(T: type);

fn F(generic T: type) {
  // The type `C(T)` is generic so that the resulting impl witness will not be
  // effectively final.
  //
  // CHECK:STDERR: fail_non_final_impl_explicit.carbon:[[@LINE+2]]:3-27: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:
  C(T) as (I where .X = ());
}

// --- concrete_query_non_final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

impl forall [U: type] U as I where .X = () {}

fn F(unused generic T: I where .X = ()) {}

fn G() {
  class C;
  F(C);
}

// --- concrete_query_non_final_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

impl forall [U: type] U as I where .X = () {}

fn F() {
class C;
  C as (I where .X = ());
}

// --- final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

final impl forall [U: type] U as I where .X = () {}

fn F(unused generic T: I where .X = ()) {}

fn G() {
  class C;
  F(C);
}

// --- final_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

final impl forall [U: type] U as I where .X = () {}

fn F() {
  class C;
  C as (I where .X = ());
}

// --- concrete_specialization_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

class C;
impl C as I where .X = () {}

fn F(unused generic T: I where .X = ()) {}
fn G() {
  F(C);
}

// --- concrete_specialization_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }

class C;
impl C as I where .X = () {}

fn F() {
  C as (I where .X = ());
}

// --- rewrite_value_in_class_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class C(T: type);

fn F(unused generic T: I where .X = C(.Y)) {}

fn G(generic T: I where .X = C(()) and .Y = ()) {
  F(T);
}

// --- fail_wrong_rewrite_value_in_class_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class C(T: type);

fn F(unused generic T: I where .X = C(.Y)) {}

fn G(generic T: I where .X = C(()) and .Y = {}) {
  // CHECK:STDERR: fail_wrong_rewrite_value_in_class_param.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I where .(I.X) = C(()) and .(I.Y) = {}` into type implementing `I where .(I.X) = C(.(I.Y))` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_wrong_rewrite_value_in_class_param.carbon:[[@LINE-4]]:21-41: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- function_in_interface_ignored.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  fn F();
}

fn F(unused generic T: I where .X = ()) {}

fn G(generic T: I where .X = ()) {
  F(T);
}

// --- function_as_rewrite_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: X;
  fn A();
}

fn F(unused generic T: I where .Y = .A) {}

fn G(generic T: I where .Y = .A) {
  F(T);
}

// --- fail_wrong_function_as_rewrite_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: X;
  fn A();
  fn B();
}

fn F(unused generic T: I where .Y = .A) {}

fn G(generic T: I where .Y = .B) {
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I where .(I.Y) = .(I.B)` into type implementing `I where .(I.Y) = .(I.A)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value.carbon:[[@LINE-4]]:21-38: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- fail_wrong_function_as_rewrite_value_in_other_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: X;
  fn A();
}

interface J {
  let J1: type;
  let J2: type;
  // B has the same index in J as A does in I, ensuring the index is not enough
  // for comparing them.
  fn B();
}

fn F(unused generic T: I & J where .Y = .A) {}

fn G(generic T: I & J where .Y = .B) {
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value_in_other_interface.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I & J where .(I.Y) = .(J.B)` into type implementing `I & J where .(I.Y) = .(I.A)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value_in_other_interface.carbon:[[@LINE-4]]:21-42: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- recursive_facet.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface K(Y: type) {  }

fn F(unused generic T: I where .X = {.k: K(I)}) {}

fn G(generic T: I where .X = {.k: K(I)}) {
  F(T);
}

// --- fail_facet_type_concrete_types_match_blanket_impl_concrete_types.carbon
library "[[@TEST_NAME]]";

interface A { let X: type; }
interface B { let Y: type; }

interface C(BB: B) { let AX: type; let BY: type; }

impl forall [AA: A, BB: B] AA as C(BB) where .AX = () and .BY = {} {}

fn F(generic AA: A where .X = (), generic BB: B where .Y = {}) {
  // The types match but there may be a specialization that specifies different
  // types and would be prefered for a specific `AA` or `BB`.
  //
  // CHECK:STDERR: fail_facet_type_concrete_types_match_blanket_impl_concrete_types.carbon:[[@LINE+2]]:3-43: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = () and .(C(BB as B).BY) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- fail_facet_type_concrete_types_become_blank_impl_types.carbon
library "[[@TEST_NAME]]";

interface A { let X: type; }
interface B { let Y: type; }

interface C(BB: B) { let AX: type; let BY: type; }

impl forall [AA: A, BB: B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(generic AA: A where .X = (), generic BB: B where .Y = {}) {
  // The types match but there may be a specialization that specifies different
  // types and would be prefered for a specific `AA` or `BB`.
  //
  // CHECK:STDERR: fail_facet_type_concrete_types_become_blank_impl_types.carbon:[[@LINE+2]]:3-43: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = () and .(C(BB as B).BY) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- facet_type_concrete_types_match_final_blanket_impl_concrete_types.carbon
library "[[@TEST_NAME]]";

interface A { let X: type; }
interface B { let Y: type; }

interface C(BB: B) { let AX: type; let BY: type; }

final impl forall [AA: A, BB: B] AA as C(BB) where .AX = () and .BY = {} {}

fn F(generic AA: A where .X = (), generic BB: B where .Y = {}) {
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- facet_type_concrete_types_become_final_blanket_impl_types.carbon
library "[[@TEST_NAME]]";

interface A { let X: type; }
interface B { let Y: type; }

interface C(BB: B) { let AX: type; let BY: type; }

final impl forall [AA: A, BB: B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(generic AA: A where .X = (), generic BB: B where .Y = {}) {
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- fail_facet_type_concrete_types_become_final_blanket_impl_types_wrong_types.carbon
library "[[@TEST_NAME]]";

interface A { let X: type; }
interface B { let Y: type; }

interface C(BB: B) { let AX: type; let BY: type; }

final impl forall [AA: A, BB: B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(generic AA: A where .X = (), generic BB: B where .Y = {}) {
  // CHECK:STDERR: fail_facet_type_concrete_types_become_final_blanket_impl_types_wrong_types.carbon:[[@LINE+2]]:3-43: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = {} and .(C(BB as B).BY) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:
  AA as (C(BB) where .AX = {} and .BY = ());
}

// --- chain_in_target.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
  let X3: type;
}

class C(T: type);

fn F(unused generic T: I where .X1 = .X3 and .X2 = C(.X3) and .X3 = ()) {}

fn G(generic T: I where .X1 = () and .X2 = C(()) and .X3 = ()) {
  F(T);
}

// --- chain_in_source.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
  let X3: type;
}

class C(T: type);

fn F(unused generic T: I where .X1 = () and .X2 = C(()) and .X3 = .X1) {}

fn G(generic T: I where .X1 = .X3 and .X2 = C(.X1) and .X3 = ()) {
  F(T);
}

// --- reference_other_facet_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}

fn F(generic U: I where .X1 = {}, unused generic T: I where .X1 = () and .X2 = U.X1) {}

fn G(generic U: I where .X1 = {} and .X2 = (), generic T: I where .X1 = U.X2 and .X2 = {}) {
  F(U, T);
}

// --- value_through_self_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: I;
  let X2: type;
  let X3: type;
}

fn F(unused generic T: I where .X1 = .Self and .X2 = .X1.X3 and .X3 = ()) {}

fn G(generic T: I where .X1 = .Self and .X2 = () and .X3 = ()) {
  F(T);
}

fn H(generic T: I where .X1 = .Self and .X2 = .X1.X3 and .X3 = ()) {
  G(T);
}

// --- rewrite_requires_subst_in_rhs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class C(T: type);

fn F(unused generic T: I where .X = C(.Y)) {}

fn G(generic T: I where .X = C({}) and .Y = {}) {
  F(T);
}

// --- facet_type_in_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}
interface J(T: type) {}

fn F(unused generic T: I where .X = J(())) {}

fn G(generic T: I where .X = J(()) and .Y = ()) {
  F(T);
}

// --- fail_facet_type_in_assoc_constant_differs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
  let Z: type;
}
interface J(T: type) {}

fn F(unused generic T: I where .X = J(())) {}

fn G(generic T: I where .X = J({}) and .Y = ()) {
  // CHECK:STDERR: fail_facet_type_in_assoc_constant_differs.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I where .(I.X) = J({}) and .(I.Y) = ()` into type implementing `I where .(I.X) = J(())` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_facet_type_in_assoc_constant_differs.carbon:[[@LINE-4]]:21-41: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- rewrite_rhs_satisfies_lhs_requirement.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {}
interface J(T: type) {
  let J1: I(T);
}

class C;
constraint L {
  // J(.Self).J1 requires that its RHS impls I(.Self), which is provided for
  // here.
  //
  // We look for this witness in `L` because the assignment is to `J(.Self).J1`
  // so we look in the facet type of `.Self` which is `L & J(.Self)`.
  require C impls I(Self);
}

fn F(unused generic T: L & J(.Self) where .J1 = C) {}

fn G(generic T: L & J(.Self) where .J1 = C) {
  F(T);
}

// --- todo_fail_rewrite_rhs_does_not_satisfy_lhs_requirement.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {}
interface J(T: type) {
  let J1: I(T);
}

class C;
constraint L {
  // Without this, C does not impl I(T), which is required by J(T).J1.
  // require C impls I(Self);
}

// TODO: This should be an error that C does not impl I(.Self). We don't yet try
// convert the RHS of a rewrite to the type of the LHS when the LHS is symbolic.
fn F(unused generic T: L & J(.Self) where .J1 = C) {}

fn G(generic T: L & J(.Self) where .J1 = C) {
  F(T);
}

// --- rewrite_requires_subst_in_nested_access_of_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
  let I2: type;
}

interface J {
  let J1: I;
}

interface K {
  let K1: J;
}

fn F(unused generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}

fn G(generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I1 = .I2) {
  F(T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
  let I2: type;
}

interface J {
  let J1: I;
}

interface K {
  let K1: J;
}

fn F(unused generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}

// F's requirement I1 = I2 is not met, as I1 = () and I2 = {}
fn G(generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I1 = () and .I2 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = () and .(I.I2) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-5]]:21-88: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// F's requirement I1 = I2 is not met, as I1 = {} and I2 is unspecified
fn G2(generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I1 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-13]]:21-88: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// F's requirement I1 = I2 is not met, as I2 = {} and I1 is unspecified
fn G3(generic T: I & J & K where .K1 = .Self and .J1 = .Self and .I2 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I2) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-21]]:21-88: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- rewrite_requires_subst_in_nested_access_of_other.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

interface J {
  let J1: I;
  let J2: type;
}

interface K {
  let K1: J;
}

// .K1.J1 is an ImplWitnessAccess into Self. The Self witness will be subst'd in
// for `U` and it will eval to the ImplWitnessAccess inside of `U.I1`. Then that
// will need to be subst'd to find the value of I1 in U's witness.
fn F(unused generic U: I, unused generic T: J & K where .J2 = (.K1.J1).I1) {}

fn G(generic U: I where .I1 = (), generic T: J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  F(U, T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

interface J {
  let J1: I;
  let J2: type;
}

interface K {
  let K1: J;
}

fn F(unused generic U: I, unused generic T: J & K where .J2 = (.K1.J1).I1) {}

// F's requirement J2 = I1 is not met as J2 = () and I1 = {}
fn G(generic U: I where .I1 = {}, generic T: J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `J & K where .(J.J2) = () and .(K.K1) = .Self as J and .(J.J1) = U as I` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-5]]:42-73: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// F's requirement J2 = I1 is not met as J2 = () and I1 is unspecified
fn G2(generic U: I, generic T: J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `J & K where .(J.J2) = () and .(K.K1) = .Self as J and .(J.J1) = U` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-13]]:42-73: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// F's requirement J2 = I1 is not met as I1 = {} and J2 is unspecified
fn G3(generic U: I where .I1 = {}, generic T: J & K where .K1 = .Self and .J1 = U) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `J & K where .(K.K1) = .Self as J and .(J.J1) = U as I` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-21]]:42-73: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// --- rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

interface J {
  let J1: type;
}

interface K {
  let K1: J & I;
  let K2: type;
  let K3: type;
}

fn F(unused generic U: I & J, unused generic T: K where .K2 = .K1.I1 and .K3 = .K1.J1) {}

fn G(generic U: I & J where .I1 = () and .J1 = {}, generic T: K where .K1 = U and .K2 = () and .K3 = {}) {
  F(U, T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

interface J {
  let J1: type;
}

interface K {
  let K1: J & I;
  let K2: type;
  let K3: type;
}

fn F(unused generic U: I & J, unused generic T: K where .K2 = .K1.I1 and .K3 = .K1.J1) {}

// K2 = I1 fails since K2 = {} and I1 = ()
fn G(generic U: I & J where .I1 = () and .J1 = {}, generic T: K where .K1 = U and .K2 = {} and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `K where .(K.K2) = {} and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-5]]:46-85: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// K2 = I1 fails since I1 = () and K2 is unspecified.
fn G2(generic U: I & J where .I1 = () and .J1 = {}, generic T: K where .K1 = U and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `K where .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-13]]:46-85: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// K2 = I1 fails since K2 = () and I1 is unspecified.
fn G3(generic U: I & J where .J1 = {}, generic T: K where .K1 = U and .K2 = () and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-21]]:46-85: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// K3 = J1 fails since J1 = {} and K3 is unspecified.
fn G4(generic U: I & J where .I1 = () and .J1 = {}, generic T: K where .K1 = U and .K2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-29]]:46-85: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// K3 = J1 fails since K3 = {} and J1 is unspecified.
fn G5(generic U: I & J where .I1 = (), generic T: K where .K1 = U and .K2 = () and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+3]]:3-9: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-37]]:46-85: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(U, T);
}

// --- fail_target_rewrites_dont_apply_to_source.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
  let I2: type;
  let I3: type;
  let I4: type;
}

fn F(unused generic T: I where .I1 = () and .I2 = ()) {}

fn G(generic T: I where .I1 = .I2) {
  // CHECK:STDERR: fail_target_rewrites_dont_apply_to_source.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `I where .(I.I1) = .(I.I2)` into type implementing `I where .(I.I1) = () and .(I.I2) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_target_rewrites_dont_apply_to_source.carbon:[[@LINE-4]]:21-52: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- nested_facet_type_used_as_root_facet_type_in_callee.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

constraint NI(V: type) {
  extend require impls I where .Y = V;
}

fn F(generic T: I where .X = NI({}), unused generic U: T.X) {}

fn G(generic T: I where .X = NI({}), generic U: I where .Y = {}) {
  F(T, U);
}

// --- fail_todo_nested_facet_type_used_as_root_facet_type_in_caller.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

constraint NI(V: type) {
  extend require impls I where .Y = V;
}

fn F(unused generic T: I where .X = NI({}), unused generic U: I where .Y = {}) {}

fn G(generic T: I where .X = NI({}), generic U: T.X) {
  // TODO: This should pass, the rewrite in the constraint `NI` should be visible.
  // CHECK:STDERR: fail_todo_nested_facet_type_used_as_root_facet_type_in_caller.carbon:[[@LINE+3]]:3-9: error: cannot convert type `U` that implements `NI({})` into type implementing `I where .(I.Y) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_todo_nested_facet_type_used_as_root_facet_type_in_caller.carbon:[[@LINE-5]]:60-77: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T, U);
}

// --- fail_nested_facet_type_used_as_root_facet_type_differs_in_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}
interface J {
  let X: type;
  let Y: type;
}

constraint NI(V: type) {
  extend require impls I where .Y = V;
}

fn F(generic T: I where .X = NI({}), unused generic U: T.X) {}

fn G(generic T: I where .X = NI({}), generic U: J where .Y = ()) {
  // CHECK:STDERR: fail_nested_facet_type_used_as_root_facet_type_differs_in_interface.carbon:[[@LINE+3]]:3-9: error: cannot convert type `U` that implements `J where .(J.Y) = ()` into type implementing `NI({})` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_nested_facet_type_used_as_root_facet_type_differs_in_interface.carbon:[[@LINE-4]]:53-58: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T, U);
}

// --- todo_fail_nested_facet_type_used_as_root_facet_type_differs_in_rewrite_callee_access.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

constraint NI(V: type) {
  extend require impls I where .Y = V;
}

fn F(generic T: I where .X = NI({}), unused generic U: T.X) {}

fn G(generic T: I where .X = NI({}), generic U: I where .Y = ()) {
  // The caller's `U` has type `I where .Y = ()` but the callee wants something
  // of type `I where .Y = {}`.
  //
  // TODO: This should fail.
  F(T, U);
}

// --- fail_nested_facet_type_used_as_root_facet_type_differs_in_rewrite_caller_access.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

constraint NI(V: type) {
  extend require impls I where .Y = V;
}

fn F(unused generic T: I where .X = NI({}), unused generic U: I where .Y = ()) {}

fn G(generic T: I where .X = NI({}), generic U: T.X) {
  // The caller's `U` has type `I where .Y = {}` but the callee wants something
  // of type `I where .Y = ()`.
  //
  // CHECK:STDERR: fail_nested_facet_type_used_as_root_facet_type_differs_in_rewrite_caller_access.carbon:[[@LINE+3]]:3-9: error: cannot convert type `U` that implements `NI({})` into type implementing `I where .(I.Y) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_nested_facet_type_used_as_root_facet_type_differs_in_rewrite_caller_access.carbon:[[@LINE-7]]:60-77: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T, U);
}

// --- period_self_in_rewrite_values.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
  let I2: type;
  let I3: type;
  let I4: type;
}

class E(T: type) {}

// * The LHS of `.I1` contains a `.Self` which is replaced by `C` to form
//   `C.(I.I1)` and resolves to `()`.
// * The RHS of `.I1` is `()` which then compares equal.
fn F1(generic _: I where .I1 = ()) {}

// * The LHS of `.I2` contains a `.Self` which is replaced by `C` to form
//   `C.(I.I1)` and resolves to `C`.
// * The RHS of `.I2` is `.Self` which is replaced by `C` and then compares
//   equal.
fn F2(generic _: I where .I2 = .Self) {}

// * The LHS of `.I3` contains a `.Self` which is replaced by `C` to form
//   `C.(I.I1)` and resolves to `E(C)`.
// * The RHS of `.I3` contains a `.Self` which is replaced by `C` to form
//   `E(C)`, and then compares equal.
fn F3(generic _: I where .I3 = E(.Self)) {}

// * The LHS of `.I4` contains a `.Self` which is replaced by `C` to form
//   `C.(I.I4)` and resolves to `.Self`. That `.Self` then is replaced by `C` so
//   the LHS untimately resolves to `C`.
// * The RHS of `.I4` is `.Self` which is replaced by `C` and then compares
//   equal.
fn F4(generic _: I where .I4 = .Self) {}

fn G() {
  class C;
  impl C as I where .I1 = () and .I2 = C and .I3 = E(C) and .I4 = .Self {}

  F1(C);
  F2(C);
  F3(C);
  F4(C);
}

// --- convert_to_period_self_preserves_self_facet_rewrite.carbon
library "[[@TEST_NAME]]";

interface J {}
interface I {
  let I1: type;
}

fn F(unused generic U: I & J where .I1 = {}) {}

impl forall [T: type] T as J {}

fn G(generic T: I where .I1 = {}) {
  // Replaces `.Self` with `T`, which converts `T` to `I & J`. Doing so has to
  // invent a witness for `J`, and this tests we do so without losing the
  // information that `.I1 = {}`.
  F(T);
}

// --- resolve_nested_impl_witness_access.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

fn G(generic _: Z where .Z2 = ()) {}

// Split the assignment of .Z1 and the use of it into separate facet types so
// that the early rewrite application doesn't get to see the value of .Z1 where
// it's used. Then rewrite constraint resolution has to do the replacement of
// .Z1 so that we know .Z2 = () as required by G.
fn F(generic T: (Z where .Z1 = ()) & (Z where .Z2 = .Z1)) {
  G(T);
}

// --- rewrite_provided_by_facet_type.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

fn F(generic _: Z where .Z1 = {}) {}

fn G(generic T: Z where .Z1 = {} and .Z2 = {}) {
  //@dump-sem-ir-begin
  // We should see a conversion happen so we know the rewrite constraint is
  // being validated.
  F(T);
  //@dump-sem-ir-end
}

// --- fail_todo_rewrite_provided_by_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

fn F(generic _: Z where .Z1 = {}) {}

constraint N {
  // The extend means that the rewrite constraint is propagated out of `N`, so
  // it can satisfy the requirement in `F` that `.Z1` is rewritten to `{}`.
  extend require impls Z where .Z1 = {};
}

fn G(generic T: N) {
  // TODO: This should pass.
  // CHECK:STDERR: fail_todo_rewrite_provided_by_named_constraint.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `N` into type implementing `Z where .(Z.Z1) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_todo_rewrite_provided_by_named_constraint.carbon:[[@LINE-11]]:14-32: note: initializing generic parameter `_` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// --- fail_rewrite_not_provided_by_named_constraint_without_extend.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

fn F(generic _: Z where .Z1 = {}) {}

constraint N {
  // Without extend, this does not actually rewrite `.Z1 = {}`, it provides the
  // equivalent of `.Z1 == {}` which allows conversion of `.Z1` to `{}` but is
  // not enough to satisfy the requirement of `F` that `.Z1` is rewritten to
  // `{}`.
  require impls Z where .Z1 = {};
}

fn G(generic T: N) {
  // CHECK:STDERR: fail_rewrite_not_provided_by_named_constraint_without_extend.carbon:[[@LINE+3]]:3-6: error: cannot convert type `T` that implements `N` into type implementing `Z where .(Z.Z1) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR: fail_rewrite_not_provided_by_named_constraint_without_extend.carbon:[[@LINE-12]]:14-32: note: initializing generic parameter `_` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(T);
}

// CHECK:STDOUT: --- rewrite_provided_by_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.b8d: <witness> = lookup_impl_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.555: type = impl_witness_access %Z.lookup_impl_witness.b8d, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.7b9: type = facet_type <@Z where %impl.elem0.555 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl.elem1.025: type = impl_witness_access %Z.lookup_impl_witness.b8d, element1 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.c91: type = facet_type <@Z where %impl.elem0.555 = %empty_struct_type and %impl.elem1.025 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %pattern_type.81e: type = pattern_type %Z_where.type.c91 [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.81e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Z_where.type.c91 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.9fe: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %facet_value: %Z_where.type.7b9 = facet_value %T.as_type, (%Z.lookup_impl_witness.9fe) [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_15.2: %Z_where.type.c91) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc14_6.2: type = facet_access_type %T.loc10_15.1 [symbolic = %T.as_type.loc14_6.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc10_15.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.9fe)]
// CHECK:STDOUT:   %facet_value.loc14_6.2: %Z_where.type.7b9 = facet_value %T.as_type.loc14_6.2, (%Z.lookup_impl_witness) [symbolic = %facet_value.loc14_6.2 (constants.%facet_value)]
// CHECK:STDOUT:   %F.specific_fn.loc14_3.2: <specific function> = specific_function constants.%F, @F(%facet_value.loc14_6.2) [symbolic = %F.specific_fn.loc14_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Z_where.type.c91 = name_ref T, %T.loc10_15.2 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc14_6.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc14_6.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc14_6.1: %Z_where.type.7b9 = facet_value %T.as_type.loc14_6.1, (constants.%Z.lookup_impl_witness.9fe) [symbolic = %facet_value.loc14_6.2 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc14: %Z_where.type.7b9 = converted %T.ref, %facet_value.loc14_6.1 [symbolic = %facet_value.loc14_6.2 (constants.%facet_value)]
// CHECK:STDOUT:     %F.specific_fn.loc14_3.1: <specific function> = specific_function %F.ref, @F(constants.%facet_value) [symbolic = %F.specific_fn.loc14_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc14_3.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
