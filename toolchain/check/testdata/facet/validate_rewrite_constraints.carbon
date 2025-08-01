// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/validate_rewrite_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/validate_rewrite_constraints.carbon

// --- facet_value.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(T:! I where .X = ()) {}

fn G(T:! I where .X = ()) {
  F(T);
}

// --- generic_interface_facet_value.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) { let X:! type; }

class C;
final impl forall [T:! type] T as I(()) where .X = () {}
final impl forall [T:! type] T as I({}) where .X = {} {}

fn F(U:! type, T:! I(U) where .X = ()) {}
fn G() {
  F((), C);
}

// --- fail_generic_interface_facet_value_wrong_specific_impl.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) { let X:! type; }

class C;
final impl forall [T:! type] T as I(()) where .X = () {}
final impl forall [T:! type] T as I({}) where .X = {} {}

fn F(U:! type, T:! I(U) where .X = ()) {}
fn G() {
  // This finds the impl where `.X = {}`, but F requires that `.X = ()`.
  //
  // CHECK:STDERR: fail_generic_interface_facet_value_wrong_specific_impl.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I({}) where .(I({}).X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F({}, C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_generic_interface_facet_value_wrong_specific_impl.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(U:! type, T:! I(U) where .X = ()) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F({}, C);
}

// --- dependent_rules.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface J { let Y:! type; }

fn F(T:! I & J where .X = .Y) {
  // Allowed since `T impls I`, `T impls J`, and has constraint providing
  // T.(I.X) = T.(J.Y)`.
  F(T);
}

// --- fail_convert.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(T:! I where .X = {.a: (), .b: {}}) {}

fn H() {
  class C;
  impl C as I where .X = {.b: {}, .a: ()} {}
  // CHECK:STDERR: fail_convert.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I where .(I.X) = {.a: (), .b: {}}` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(C);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_convert.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = {.a: (), .b: {}}) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(C);
}

fn G(T:! I where .X = {.b: {}, .a: ()}) {
  // CHECK:STDERR: fail_convert.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = {.b: {}, .a: ()}` into type implementing `I where .(I.X) = {.a: (), .b: {}}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_convert.carbon:[[@LINE-19]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = {.a: (), .b: {}}) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_todo_dependent_rules_compound.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface J { let Y:! type; }

// CHECK:STDERR: fail_todo_dependent_rules_compound.carbon:[[@LINE+8]]:23: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: fn F(T:! I & J where .(I.X) = .(J.Y)) {
// CHECK:STDERR:                       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_dependent_rules_compound.carbon:[[@LINE+4]]:23: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn F(T:! I & J where .(I.X) = .(J.Y)) {
// CHECK:STDERR:                       ^
// CHECK:STDERR:
fn F(T:! I & J where .(I.X) = .(J.Y)) {
  // Allowed since `T impls I`, `T impls J`, and has constraint providing
  // T.(I.X) = T.(J.Y)`.
  F(T);
}

// --- parameterized_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) { let X:! type; }

final impl forall [J:! I(()) where .X = ()] J as I({}) where .X = {} {}

fn F(T:! I({}) where .X = {}) {}

fn G(T:! I(()) where .X = ()) {
  F(T);
}

// --- fail_todo_parameterized_interface_compound.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) { let X:! type; }

final impl forall [J:! I(())] J as I({}) where .X = {} {}

// CHECK:STDERR: fail_todo_parameterized_interface_compound.carbon:[[@LINE+8]]:31: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: fn F(T:! I(()) & I({}) where .(I(()).X) = () and .(I({}).X) = {}) {}
// CHECK:STDERR:                               ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_parameterized_interface_compound.carbon:[[@LINE+4]]:31: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn F(T:! I(()) & I({}) where .(I(()).X) = () and .(I({}).X) = {}) {}
// CHECK:STDERR:                               ^
// CHECK:STDERR:
fn F(T:! I(()) & I({}) where .(I(()).X) = () and .(I({}).X) = {}) {}

fn G(T:! I(()) where .X = ()) {
  F(T);
}

// --- fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) { let X:! type; }

final impl forall [J:! I(()) where .X = {}] J as I({}) where .X = {} {}

fn F(T:! I({}) where .X = {}) {}

fn G(T:! I(()) where .X = ()) {
  // CHECK:STDERR: fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I(()) where .(I(()).X) = ()` into type implementing `I({}) where .(I({}).X) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_parameterized_interface_with_wrong_where_in_impl_deduction.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I({}) where .X = {}) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- source_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

fn F(T:! I where .X1 = () and .X2 = ()) {}

fn G() {
  class C;
  impl C as I where .X1 = .X2 and .X2 = () {}
  F(C);
}

fn H(T:! I where .X1 = .X2 and .X2 = ()) {
  F(T);
}

// --- target_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

fn F(T:! I where .X1 = .X2 and .X2 = ()) {}

fn G() {
  class C;
  impl C as I where .X1 = () and .X2 = () {}
  F(C);
}

fn H(T:! I where .X1 = () and .X2 = ()) {
  F(T);
}

// --- both_rhs_is_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

fn F(T:! I where .X1 = .X2 and .X2 = ()) {}

fn G() {
  class C1;
  impl C1 as I where .X1 = .X2 and .X2 = () {}
  F(C1);

  class C2;
  impl C2 as I where .X1 = () and .X2 = .X1 {}
  F(C2);
}

fn H1(T:! I where .X1 = .X2 and .X2 = ()) {
  F(T);
}

fn H2(T:! I where .X1 = () and .X2 = .X1) {
  F(T);
}

// --- fail_error_in_witness_table.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

fn F(T:! I where .X1 = .X2) {}

class C;
// .X2 is not set in the impl definition, so its value is an ErrorInst
// in the impl's witness table.
//
// CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE+7]]:1: error: associated constant X2 not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl C as I where .X1 = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE-12]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let X2:! type;
// CHECK:STDERR:       ^~~~~~~~~
// CHECK:STDERR:
impl C as I where .X1 = () {}

fn G() {
  // CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I where .(I.X1) = .(I.X2)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(C);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_error_in_witness_table.carbon:[[@LINE-19]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X1 = .X2) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(C);
}

// --- impl_provides_rewrite_requirements.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J { let Y:! type; }

class C;

final impl forall [T:! I] T as J where .Y = C {}

fn F(T:! I & J where .Y = C) {}
fn G(T:! I) {
  F(T);
}

// --- facet_provides_rewrite_requirements.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J { let Y:! type; }

class C;

final impl forall [T:! J] T as I {}

fn F(T:! I & J where .Y = C) {}
fn G(T:! J where .Y = C) {
  F(T);
}

// --- fail_non_final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

impl forall [U:! type] U as I where .X = () {}

fn F(T:! I where .X = ()) {}

class C(T:! type);

fn G(T:! type) {
  // The type `C(T)` is generic so that the resulting impl witness will not be
  // effectively final.
  //
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE+7]]:3: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(C(T));
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE-11]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = ()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(C(T));
}

// --- fail_non_final_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

impl forall [U:! type] U as I where .X = () {}

class C(T:! type);

fn F(T:! type) {
  // The type `C(T)` is generic so that the resulting impl witness will not be
  // effectively final.
  //
  // CHECK:STDERR: fail_non_final_impl_explicit.carbon:[[@LINE+4]]:3: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(T) as (I where .X = ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  C(T) as (I where .X = ());
}

// --- concrete_query_non_final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

impl forall [U:! type] U as I where .X = () {}

fn F(T:! I where .X = ()) {}

fn G() {
  class C;
  F(C);
}

// --- concrete_query_non_final_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

impl forall [U:! type] U as I where .X = () {}

fn F() {
class C;
  C as (I where .X = ());
}

// --- final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

final impl forall [U:! type] U as I where .X = () {}

fn F(T:! I where .X = ()) {}

fn G() {
  class C;
  F(C);
}

// --- final_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

final impl forall [U:! type] U as I where .X = () {}

fn F() {
  class C;
  C as (I where .X = ());
}

// --- concrete_specialization_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
impl C as I where .X = () {}

fn F(T:! I where .X = ()) {}
fn G() {
  F(C);
}

// --- concrete_specialization_impl_explicit_as.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
impl C as I where .X = () {}

fn F() {
  C as (I where .X = ());
}

// --- rewrite_value_in_class_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

class C(T:! type);

fn F(T:! I where .X = C(.Y)) {}

fn G(T:! I where .X = C(()) and .Y = ()) {
  F(T);
}

// --- fail_wrong_rewrite_value_in_class_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

class C(T:! type);

fn F(T:! I where .X = C(.Y)) {}

fn G(T:! I where .X = C(()) and .Y = {}) {
  // CHECK:STDERR: fail_wrong_rewrite_value_in_class_param.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = C(()) and .(I.Y) = {}` into type implementing `I where .(I.X) = C(.(I.Y))` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_wrong_rewrite_value_in_class_param.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = C(.Y)) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- function_in_interface_ignored.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn F();
}

fn F(T:! I where .X = ()) {}

fn G(T:! I where .X = ()) {
  F(T);
}

// --- function_as_rewrite_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! X;
  fn A();
}

fn F(T:! I where .Y = .A) {}

fn G(T:! I where .Y = .A) {
  F(T);
}

// --- fail_wrong_function_as_rewrite_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! X;
  fn A();
  fn B();
}

fn F(T:! I where .Y = .A) {}

fn G(T:! I where .Y = .B) {
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.Y) = .(I.B)` into type implementing `I where .(I.Y) = .(I.A)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .Y = .A) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_wrong_function_as_rewrite_value_in_other_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! X;
  fn A();
}

interface J {
  let J1:! type;
  let J2:! type;
  // B has the same index in J as A does in I, ensuring the index is not enough
  // for comparing them.
  fn B();
}

fn F(T:! I & J where .Y = .A) {}

fn G(T:! I & J where .Y = .B) {
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value_in_other_interface.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I & J where .(I.Y) = .(J.B)` into type implementing `I & J where .(I.Y) = .(I.A)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_wrong_function_as_rewrite_value_in_other_interface.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I & J where .Y = .A) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- recursive_facet.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface K(Y:! type) {  }

fn F(T:! I where .X = {.k: K(I where .X = ())}) {}

fn G(T:! I where .X = {.k: K(I where .X = ())}) {
  F(T);
}

// --- fail_facet_type_concrete_types_match_blanket_impl_concrete_types.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

interface C(BB:! B) { let AX:! type; let BY:! type; }

impl forall [AA:! A, BB:! B] AA as C(BB) where .AX = () and .BY = {} {}

fn F(AA:! A where .X = (), BB:! B where .Y = {}) {
  // The types match but there may be a specialization that specifies different
  // types and would be prefered for a specific `AA` or `BB`.
  //
  // CHECK:STDERR: fail_facet_type_concrete_types_match_blanket_impl_concrete_types.carbon:[[@LINE+4]]:3: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = () and .(C(BB as B).BY) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   AA as (C(BB) where .AX = () and .BY = {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- fail_facet_type_concrete_types_become_blank_impl_types.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

interface C(BB:! B) { let AX:! type; let BY:! type; }

impl forall [AA:! A, BB:! B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(AA:! A where .X = (), BB:! B where .Y = {}) {
  // The types match but there may be a specialization that specifies different
  // types and would be prefered for a specific `AA` or `BB`.
  //
  // CHECK:STDERR: fail_facet_type_concrete_types_become_blank_impl_types.carbon:[[@LINE+4]]:3: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = () and .(C(BB as B).BY) = {}` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   AA as (C(BB) where .AX = () and .BY = {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- facet_type_concrete_types_match_final_blanket_impl_concrete_types.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

interface C(BB:! B) { let AX:! type; let BY:! type; }

final impl forall [AA:! A, BB:! B] AA as C(BB) where .AX = () and .BY = {} {}

fn F(AA:! A where .X = (), BB:! B where .Y = {}) {
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- facet_type_concrete_types_become_final_blanket_impl_types.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

interface C(BB:! B) { let AX:! type; let BY:! type; }

final impl forall [AA:! A, BB:! B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(AA:! A where .X = (), BB:! B where .Y = {}) {
  AA as (C(BB) where .AX = () and .BY = {});
}

// --- fail_facet_type_concrete_types_become_final_blanket_impl_types_wrong_types.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

interface C(BB:! B) { let AX:! type; let BY:! type; }

final impl forall [AA:! A, BB:! B] AA as C(BB) where .AX = AA.X and .BY = BB.Y {}

fn F(AA:! A where .X = (), BB:! B where .Y = {}) {
  // CHECK:STDERR: fail_facet_type_concrete_types_become_final_blanket_impl_types_wrong_types.carbon:[[@LINE+4]]:3: error: cannot convert type `AA` that implements `A where .(A.X) = ()` into type implementing `C(BB as B) where .(C(BB as B).AX) = {} and .(C(BB as B).BY) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   AA as (C(BB) where .AX = {} and .BY = ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AA as (C(BB) where .AX = {} and .BY = ());
}

// --- chain_in_target.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
}

class C(T:! type);

fn F(T:! I where .X1 = .X3 and .X2 = C(.X3) and .X3 = ()) {}

fn G(T:! I where .X1 = () and .X2 = C(()) and .X3 = ()) {
  F(T);
}

// --- chain_in_source.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
}

class C(T:! type);

fn F(T:! I where .X1 = () and .X2 = C(()) and .X3 = .X1) {}

fn G(T:! I where .X1 = .X3 and .X2 = C(.X1) and .X3 = ()) {
  F(T);
}

// --- reference_other_facet_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

fn F(U:! I where .X1 = {}, T:! I where .X1 = () and .X2 = U.X1) {}

fn G(U:! I where .X1 = {} and .X2 = (), T:! I where .X1 = U.X2 and .X2 = {}) {
  F(U, T);
}

// --- fail_todo_value_through_self_value.carbon
library "[[@TEST_NAME]]";

interface I {
  // TODO: This should not be diagnosed.
  //
  // CHECK:STDERR: fail_todo_value_through_self_value.carbon:[[@LINE+7]]:12: error: associated constant has incomplete type `I` [IncompleteTypeInAssociatedConstantDecl]
  // CHECK:STDERR:   let X1:! I;
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_todo_value_through_self_value.carbon:[[@LINE-6]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface I {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let X1:! I;
  let X2:! type;
  let X3:! type;
}

fn F(T:! I where .X1 = .Self and .X2 = .X1.X3 and .X3 = ());

fn G(T:! I where .X1 = .Self and .X2 = () and .X3 = ()) {
  F(T);
}

fn H(T:! I where .X1 = .Self and .X2 = .X1.X3 and .X3 = ()) {
  G(T);
}

// --- associated_constant_is_facet_type_of_same_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let A:! type;
  let X:! type;
}

class C;

// This looks for a bug where `.Self.A` resolves to `C` from `.T.A` in the
// incoming facet value, which is incorrect. It should be `.T.X.A` which
// resolves to `{}`.
fn F(U:! I where .X = (I where .A = {})) {}

fn G(T:! I where .X = (I where .A = {}) and .A = C) {
  F(T);
}

// --- rewrite_requires_subst_in_rhs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

class C(T:! type);

fn F(T:! I where .X = C(.Y)) {}

fn G(T:! I where .X = C({}) and .Y = {}) {
  F(T);
}

// --- fail_todo_rewrite_requires_subst_in_nested_facet_type.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  let X:! type;
  let Y:! type;
}

class C;

fn F(T:! I(C) where .X = (I(.Y) where .Y = ())) {}

fn G(T:! I(C) where .X = (I({}) where .Y = ()) and .Y = {}) {
  // TODO: The T in G should match the T in F, once the .Self reference to the
  // top level facet value in `I(.Y)` is correctly substituted by tracking that
  // it is a .Self reference to the top level Self.
  // CHECK:STDERR: fail_todo_rewrite_requires_subst_in_nested_facet_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I(C) where .(I(C).X) = I({}) where .(I({}).Y) = () and .(I(C).Y) = {}` into type implementing `I(C) where .(I(C).X) = I(.(I(C).Y)) where .(I(.(I(C).Y)).Y) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_rewrite_requires_subst_in_nested_facet_type.carbon:[[@LINE-9]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I(C) where .X = (I(.Y) where .Y = ())) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

fn G2(T:! I(C) where .X = (I(.Y) where .Y = ()) and .Y = {}) {
  F(T);
}

// --- fail_rewrite_requires_subst_in_nested_facet_type_types_differ.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  let X:! type;
  let Y:! type;
}

fn F(T:! I({}) where .X = (I(.Y) where .X = ())) {}

// I(.Y) is I({}) which doesn't match I(()).
fn G(T:! I({}) where .X = (I(()) where .X = ()) and .Y = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_facet_type_types_differ.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I({}) where .(I({}).X) = I(()) where .(I(()).X) = () and .(I({}).Y) = {}` into type implementing `I({}) where .(I({}).X) = I(.(I({}).Y)) where .(I(.(I({}).Y)).X) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_facet_type_types_differ.carbon:[[@LINE-7]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I({}) where .X = (I(.Y) where .X = ())) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- facet_type_in_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn F(T:! I where .X = (I where .X = () and .Y = ())) {}

fn G(T:! I where .X = (I where .X = .Y and .Y = ())) {
  F(T);
}

// --- fail_facet_type_in_assoc_constant_differs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
  let Z:! type;
}

fn F(T:! I where .X = (I where .X = .Y)) {}

fn G(T:! I where .X = (I where .X = () and .Y = () and .Z = {})) {
  // CHECK:STDERR: fail_facet_type_in_assoc_constant_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.X) = () and .(I.Y) = () and .(I.Z) = {}` into type implementing `I where .(I.X) = I where .(I.X) = .(I.Y)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_facet_type_in_assoc_constant_differs.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = (I where .X = .Y)) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- nested_facet_type_in_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
  let Z:! type;
}

fn F(T:! I where .X = (I where .Y = (I where .X = () and .Y = ()))) {}

fn G(T:! I where .X = (I where .Y = (I where .X = .Y and .Y = ()))) {
  F(T);
}

// --- fail_nested_facet_type_assigns_same_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn F(T:! I where .Y = ()) {}

// The `.Y = ()` is on a different `.Self` than `T` (an unattached self), so
// should not satisfy `F`.
fn G(T:! I where .X = (I where .Y = ())) {
  // CHECK:STDERR: fail_nested_facet_type_assigns_same_assoc_constant.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = ()` into type implementing `I where .(I.Y) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_nested_facet_type_assigns_same_assoc_constant.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .Y = ()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_nested_facet_type_in_assoc_constant_differs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
  let Z:! type;
}

fn F(T:! I where .X = (I where .Y = (I where .X = .Y))) {}

// `.X = .Y` does not match `.X = () and .Y = ()` as they are different resolved
// facet types.
fn G(T:! I where .X = (I where .Y = (I where .X = () and .Y = ()))) {
  // CHECK:STDERR: fail_nested_facet_type_in_assoc_constant_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = I where .(I.X) = () and .(I.Y) = ()` into type implementing `I where .(I.X) = I where .(I.Y) = I where .(I.X) = .(I.Y)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_nested_facet_type_in_assoc_constant_differs.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y))) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// The extra .Z rewrite makes a different resolved facet type which does not
// match.
fn G2(T:! I where .X = (I where .Y = (I where .X = .Y and .Z = {}))) {
  // CHECK:STDERR: fail_nested_facet_type_in_assoc_constant_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = I where .(I.X) = .(I.Y) and .(I.Z) = {}` into type implementing `I where .(I.X) = I where .(I.Y) = I where .(I.X) = .(I.Y)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_nested_facet_type_in_assoc_constant_differs.carbon:[[@LINE-21]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y))) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_nested_facet_type_from_constant.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}

// References to named constants in a facet type don't work at all. If they did,
// then when the `Constant` facet type is put into the RHS of a rewrite
// constraint, its references to `.Self` must be modified to not refer to the
// top level `.Self` which is `T`. If done correctly, they will match the
// `.Self` references in the same position in the parameter of `F`. If not, the
// `.X` within becomes self-referential and makes a cycle.

fn G1() {
  let Constant:! type = I where .X = .Y;

  fn G(T:! I where .X = (I where .Y = (Constant, ))) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = (Constant,)` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-16]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

fn G2() {
  let Constant:! type = (I where .X = .Y, );

  fn G(T:! I where .X = (I where .Y = Constant)) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = Constant` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-31]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

fn G3() {
  let Constant:! type = I where .Y = (I where .X = .Y, );

  fn G(T:! I where .X = Constant) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = Constant` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-46]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

fn G4() {
  let Constant2:! type = (I where .X = .Y, );
  let Constant:! type = Constant2;

  fn G(T:! I where .X = (I where .Y = Constant)) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = Constant` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-62]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

fn G5() {
  let Constant2:! type = I where .X = .Y;
  let Constant:! type = (Constant2, );

  fn G(T:! I where .X = (I where .Y = Constant)) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = Constant` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-78]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

fn G6() {
  let Constant2:! type = (I where .X = .Y, );
  let Constant:! type = I where .Y = Constant2;

  fn G(T:! I where .X = Constant) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = Constant` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant.carbon:[[@LINE-94]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

// ---fail_nested_facet_type_from_constant_differs.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}

fn G1() {
  let Constant:! type = I where .X = () and .Y = ();

  fn G(T:! I where .X = (I where .Y = (Constant, ))) {
    // CHECK:STDERR: fail_nested_facet_type_from_constant_differs.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = I where .(I.Y) = (Constant,)` into type implementing `I where .(I.X) = I where .(I.Y) = (I where .(I.X) = .(I.Y),)` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_nested_facet_type_from_constant_differs.carbon:[[@LINE-9]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: fn F(T:! I where .X = (I where .Y = (I where .X = .Y, ))) {}
    // CHECK:STDERR:      ^
    // CHECK:STDERR:
    F(T);
  }
}

// --- rewrite_requires_subst_in_nested_access_of_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
  let I2:! type;
}

interface J {
  let J1:! I;
}

interface K {
  let K1:! J;
}

fn F(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}

fn G(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = .I2) {
  F(T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
  let I2:! type;
}

interface J {
  let J1:! I;
}

interface K {
  let K1:! J;
}

fn F(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}

// F's requirement I1 = I2 is not met, as I1 = () and I2 = {}
fn G(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = () and .I2 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = () and .(I.I2) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-7]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// F's requirement I1 = I2 is not met, as I1 = {} and I2 is unspecified
fn G2(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-19]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// F's requirement I1 = I2 is not met, as I2 = {} and I1 is unspecified
fn G3(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I2 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I2) = {}` into type implementing `I & J & K where .(K.K1) = .Self as J and .(J.J1) = .Self as I and .(I.I1) = .(I.I2)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_self_wrong_type.carbon:[[@LINE-31]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I & J & K where .K1 = .Self and .J1 = .Self and .I1 = (.K1.J1).I2) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_todo_rewrite_requires_subst_in_nested_access_of_other.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

interface J {
  let J1:! I;
  let J2:! type;
}

interface K {
  let K1:! J;
}

// .K1.J1 is an ImplWitnessAccess into Self. The Self witness will be subst'd in
// for `U` and it will eval to the ImplWitnessAccess inside of `U.I1`. Then that
// will need to be subst'd to find the value of I1 in U's witness.
fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}

fn G(U:! I where .I1 = (), T:! J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  // CHECK:STDERR: fail_todo_rewrite_requires_subst_in_nested_access_of_other.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `J & K where .(J.J2) = () and .(K.K1) = .Self as J and .(J.J1) = U as I` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_rewrite_requires_subst_in_nested_access_of_other.carbon:[[@LINE-6]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  F(U, T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

interface J {
  let J1:! I;
  let J2:! type;
}

interface K {
  let K1:! J;
}

fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}

// F's requirement J2 = I1 is not met as J2 = () and I1 = {}
fn G(U:! I where .I1 = {}, T:! J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `J & K where .(J.J2) = () and .(K.K1) = .Self as J and .(J.J1) = U as I` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-7]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  F(U, T);
}

// F's requirement J2 = I1 is not met as J2 = () and I1 is unspecified
fn G2(U:! I, T:! J & K where .K1 = .Self and .J1 = U and .J2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `J & K where .(J.J2) = () and .(K.K1) = .Self as J and .(J.J1) = U` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-19]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  F(U, T);
}

// F's requirement J2 = I1 is not met as I1 = {} and J2 is unspecified
fn G3(U:! I where .I1 = {}, T:! J & K where .K1 = .Self and .J1 = U) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `J & K where .(K.K1) = .Self as J and .(J.J1) = U as I` into type implementing `J & K where .(J.J2) = .(K.K1).(J.J1).(I.I1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_wrong_type.carbon:[[@LINE-31]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I, T:! J & K where .J2 = (.K1.J1).I1) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  F(U, T);
}

// --- rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

interface J {
  let J1:! type;
}

interface K {
  let K1:! J & I;
  let K2:! type;
  let K3:! type;
}

fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}

fn G(U:! I & J where .I1 = () and .J1 = {}, T:! K where .K1 = U and .K2 = () and .K3 = {}) {
  F(U, T);
}

// --- fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

interface J {
  let J1:! type;
}

interface K {
  let K1:! J & I;
  let K2:! type;
  let K3:! type;
}

fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}

// K2 = I1 fails since K2 = {} and I1 = ()
fn G(U:! I & J where .I1 = () and .J1 = {}, T:! K where .K1 = U and .K2 = {} and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `K where .(K.K2) = {} and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-7]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F(U, T);
}

// K2 = I1 fails since I1 = () and K2 is unspecified.
fn G2(U:! I & J where .I1 = () and .J1 = {}, T:! K where .K1 = U and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `K where .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-19]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F(U, T);
}

// K2 = I1 fails since K2 = () and I1 is unspecified.
fn G3(U:! I & J where .J1 = {}, T:! K where .K1 = U and .K2 = () and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-31]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F(U, T);
}

// K3 = J1 fails since J1 = {} and K3 is unspecified.
fn G4(U:! I & J where .I1 = () and .J1 = {}, T:! K where .K1 = U and .K2 = ()) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-43]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F(U, T);
}

// K3 = J1 fails since K3 = {} and J1 is unspecified.
fn G5(U:! I & J where .I1 = (), T:! K where .K1 = U and .K2 = () and .K3 = {}) {
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `K where .(K.K2) = () and .(K.K1) = U as I & J and .(K.K3) = {}` into type implementing `K where .(K.K2) = .(K.K1).(I.I1) and .(K.K3) = .(K.K1).(J.J1)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(U, T);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_rewrite_requires_subst_in_nested_access_of_other_with_two_witnesses_wrong_type.carbon:[[@LINE-55]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! I & J, T:! K where .K2 = .K1.I1 and .K3 = .K1.J1) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F(U, T);
}

// --- fail_target_rewrites_dont_apply_to_source.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
  let I2:! type;
  let I3:! type;
  let I4:! type;
}

fn F(T:! I where .I1 = () and .I2 = ()) {}

fn G(T:! I where .I1 = .I2) {
  // CHECK:STDERR: fail_target_rewrites_dont_apply_to_source.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.I1) = .(I.I2)` into type implementing `I where .(I.I1) = () and .(I.I2) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_target_rewrites_dont_apply_to_source.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .I1 = () and .I2 = ()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- nested_facet_type_used_as_root_facet_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn F(T:! I where .X = (I where .Y = {}), U:! T.X) {}

fn G(T:! I where .X = (I where .Y = {}), U:! I where .Y = {}) {
  F(T, U);
}
