// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/validate_rewrite_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/validate_rewrite_constraints.carbon

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

final impl forall [J:! I(())] J as I({}) where .X = {} {}

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

// --- dependent_rules_chains_from_facet_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
  // Only used in source facet, to ensure the facet type differs from the
  // conversion target.
  let Y:! type;
}

fn F1(T:! I where .X1 = .X2 and .X2 = .X3 and .X2 = ()) {}
fn F2(T:! I where .X1 = .X3 and .X2 = .X3 and .X2 = ()) {}

fn G1(T:! I where .X1 = () and .X2 = () and .X3 = () and .Y = ()) {
  F1(T);
  F2(T);
}

fn G2(T:! I where .X1 = () and .X2 = () and .X3 = .X2 and .Y = ()) {
  F1(T);
  F2(T);
}

fn G3(T:! I where .X1 = .X3 and .X2 = () and .X3 = .X2 and .Y = ()) {
  F1(T);
  F2(T);
}

fn G4(T:! I where .X1 = () and .X2 = .X3 and .X3 = .X1 and .Y = ()) {
  F1(T);
  F2(T);
}

// .X2 and .X3 are equal but have no known value.
fn F3(T:! I where .X1 = () and .X2 = .X3 and .X3 = .X2) {}

fn G5(T:! I where .X3 = .X2 and .X2 = .X3 and .X1 = () and .Y = ()) {
  F3(T);
}

fn G6(T:! I where .X3 = .X3 and .X2 = .X3 and .X1 = () and .Y = ()) {
  F3(T);
}

// .X2 is assigned itself, which precludes it from having a known value.
fn F4(T:! I where .X1 = () and .X2 = .X2 and .X3 = .X2) {}

fn G7(T:! I where .X3 = .X2 and .X2 = .X3 and .X1 = () and .Y = ()) {
  F4(T);
}

fn G8(T:! I where .X3 = .X3 and .X2 = .X3 and .X1 = () and .Y = ()) {
  F4(T);
}

// --- dependent_rules_chains_from_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
  // Only used in source facet, to ensure the facet type differs from the
  // conversion target.
  let Y:! type;
}

fn F1(T:! I where .X1 = .X2 and .X2 = .X3 and .X2 = ()) {}
fn F2(T:! I where .X1 = .X3 and .X2 = .X3 and .X2 = ()) {}

class C1;
impl C1 as I where .X1 = () and .X2 = () and .X3 = () and .Y = () {}

fn G1() {
  F1(C1);
  F2(C1);
}

class C2;
impl C2 as I where .X1 = () and .X2 = () and .X3 = .X2 and .Y = () {}

fn G2() {
  F1(C2);
  F2(C2);
}

class C3;
impl C3 as I where .X1 = .X3 and .X2 = () and .X3 = .X2 and .Y = () {}

fn G3() {
  F1(C3);
  F2(C3);
}

class C4;
impl C4 as I where .X1 = () and .X2 = .X3 and .X3 = .X1 and .Y = () {}

fn G4() {
  F1(C4);
  F2(C4);
}

// .X2 and .X3 are equal but have no known value.
fn F3(T:! I where .X1 = () and .X2 = .X3 and .X3 = .X2) {}

class C5;
impl C5 as I where .X3 = .X2 and .X2 = .X3 and .X1 = () and .Y = () {}

fn G5() {
  F3(C5);
}

class C6;
impl C6 as I where .X3 = .X3 and .X2 = .X3 and .X1 = () and .Y = () {}

fn G6() {
  F3(C6);
}

// .X2 is assigned itself, which precludes it from having a known value.
fn F4(T:! I where .X1 = () and .X2 = .X2 and .X3 = .X2) {}

class C7;
impl C7 as I where .X3 = .X2 and .X2 = .X3 and .X1 = () and .Y = () {}

fn G7() {
  F4(C7);
}

class C8;
impl C8 as I where .X3 = .X3 and .X2 = .X3 and .X1 = () and .Y = () {}

fn G8() {
  F4(C8);
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

class C(T:! type);
impl forall [T:! type] T as I where .X = () {}

fn F(T:! I where .X = ()) {}
fn G(T:! type) {
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE+7]]:3: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(C(T));
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_non_final_impl_deduce.carbon:[[@LINE-5]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = ()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(C(T));
}

// --- fail_non_final_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C(T:! type);
impl forall [T:! type] T as I where .X = () {}

fn F(T:! type) {
  // CHECK:STDERR: fail_non_final_impl_explicit.carbon:[[@LINE+4]]:3: error: cannot convert type `C(T)` into type implementing `I where .(I.X) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(T) as (I where .X = ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  C(T) as (I where .X = ());
}

// --- concrete_query_non_final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
impl forall [T:! type] T as I where .X = () {}

fn F(T:! I where .X = ()) {}
fn G() {
  F(C);
}

// --- concrete_query_non_final_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
impl forall [T:! type] T as I where .X = () {}

fn F() {
  C as (I where .X = ());
}

// --- final_impl_deduce.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
final impl forall [T:! type] T as I where .X = () {}

fn F(T:! I where .X = ()) {}
fn G() {
  F(C);
}

// --- final_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
final impl forall [T:! type] T as I where .X = () {}

fn F() {
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

// --- concrete_specialization_impl_explicit.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

class C;
impl C as I where .X = () {}

fn F() {
  C as (I where .X = ());
}

// --- todo_rewrite_value_in_class_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

class C(T:! type);

fn F(T:! I where .X = C(.Y)) {}

fn G(T:! I where .X = C(()) and .Y = ()) {
  // CHECK:STDERR: todo_rewrite_value_in_class_param.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = C(()) and .(I.Y) = ()` into type implementing `I where .(I.X) = C(.(I.Y))` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: todo_rewrite_value_in_class_param.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = C(.Y)) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
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

// --- todo_fail_wrong_function_as_rewrite_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! X;
  fn A();
  fn B();
}

fn F(T:! I where .Y = .A) {}

fn G(T:! I where .Y = .B) {
  // TODO: This should be diagnosed as the wrong value being assigned to .Y for
  // what F is expecting.
  //
  F(T);
}
