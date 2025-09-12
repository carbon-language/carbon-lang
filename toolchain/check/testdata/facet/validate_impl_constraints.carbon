// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon

// --- self_impls_modifies_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(T:! I where .X = ()) {}

fn G(T:! I where .Self impls (I where .X = ())) {
  F(T);
}

// --- fail_self_impls_modifies_assoc_constant_type_differs.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }

fn F(T:! I where .X = ()) {}

fn G(T:! I where .Self impls (I where .X = {})) {
  // CHECK:STDERR: fail_self_impls_modifies_assoc_constant_type_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `I where .(I.X) = {}` into type implementing `I where .(I.X) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_self_impls_modifies_assoc_constant_type_differs.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where .X = ()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon
library "[[@TEST_NAME]]";

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M { let M0:! type; }
impl forall [U:! L] C(U) as M where .M0 = {} {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(U:! type where C(.Self) impls (M where .M0 = {})) {}

fn G(T:! L) {
  // CHECK:STDERR: fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where...` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon:[[@LINE-6]]:6: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! type where C(.Self) impls (M where .M0 = {})) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon
library "[[@TEST_NAME]]";

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M { let M0:! type; }
impl forall [U:! L] C(U) as M where .M0 = () {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(U:! type where C(.Self) impls (M where .M0 = {})) {}

fn G(T:! L) {
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where...` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_where_impls_tests_associated_constant_of_generic_type_type_differs.carbon:[[@LINE-6]]:6: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! type where C(.Self) impls (M where .M0 = {})) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// --- self_in_interface_generic_param_unconstrained.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface I(T:! type) {}

fn F(T:! I(.Self) where .Self impls Z) {}

fn G(T:! Z & I(.Self)) {
  F(T);
}

// --- fail_todo_self_in_interface_generic_param_constrained.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface I(T:! Z) {}

// Implied constraint: .Self impls Z, which is satisfied and checked at the end
// of the fn signature.
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE+7]]:10: error: cannot convert type `.Self` that implements `type` into type implementing `Z` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn F(T:! I(.Self) where .Self impls Z) {}
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE-7]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface I(T:! Z) {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn F(T:! I(.Self) where .Self impls Z) {}

// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE+7]]:14: error: cannot convert type `.Self` that implements `type` into type implementing `Z` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn G(T:! Z & I(.Self)) {
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR: fail_todo_self_in_interface_generic_param_constrained.carbon:[[@LINE-16]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface I(T:! Z) {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn G(T:! Z & I(.Self)) {
  F(T);
}
