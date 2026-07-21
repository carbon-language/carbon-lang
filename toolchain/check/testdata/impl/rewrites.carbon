// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/rewrites.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/rewrites.carbon

// --- rewrite_in_facet_type_before.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

impl () as J {}

// This uses the earlier rewrite to see that we need `() impls J`.
final impl forall [T: type] T as I where .X = () and .X impls J {}

// --- fail_rewrite_in_facet_type_before_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_facet_type_before_missing_impl.carbon:[[@LINE+4]]:1: error: constraint `I where () impls J and .(I.X) = ()` being implemented requires that `()` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as I where .X = () and .X impls J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as I where .X = () and .X impls J {}

// --- fail_todo_rewrite_in_facet_type_after.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

impl () as J {}

// We don't see the rewrite until later, but when we look at the constraints
// required, we still need to see that we need `() impls J`.
// CHECK:STDERR: fail_todo_rewrite_in_facet_type_after.carbon:[[@LINE+4]]:1: error: constraint `I where T.(I.X) impls J and .(I.X) = ()` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as I where .X impls J and .X = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as I where .X impls J and .X = () {}

// --- fail_rewrite_in_facet_type_after_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_facet_type_after_missing_impl.carbon:[[@LINE+4]]:1: error: constraint `I where T.(I.X) impls J and .(I.X) = ()` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as I where .X impls J and .X = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as I where .X impls J and .X = () {}

// --- fail_todo_rewrite_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
}

impl () as J {}

// The rewrite from `extend` is inherited and should satisfy that `.X impls J`.
// CHECK:STDERR: fail_todo_rewrite_in_named_constraint.carbon:[[@LINE+11]]:1: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: final impl forall [T: type] T as GivesI where .X impls J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_rewrite_in_named_constraint.carbon:[[@LINE-13]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface I { let X: type; }
// CHECK:STDERR:                   ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_rewrite_in_named_constraint.carbon:[[@LINE+4]]:1: error: constraint `GivesI where T.(I.X) impls J` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as GivesI where .X impls J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as GivesI where .X impls J {}

// --- fail_rewrite_in_named_constraint_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_named_constraint_missing_impl.carbon:[[@LINE+11]]:1: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: final impl forall [T: type] T as GivesI where .X impls J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_rewrite_in_named_constraint_missing_impl.carbon:[[@LINE-11]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface I { let X: type; }
// CHECK:STDERR:                   ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_rewrite_in_named_constraint_missing_impl.carbon:[[@LINE+4]]:1: error: constraint `GivesI where T.(I.X) impls J` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as GivesI where .X impls J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as GivesI where .X impls J {}

// --- fail_todo_rewrite_and_impls_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
  require impls I where .X impls J;
}

impl () as J {}

// The rewrite from `extend` is inherited and should satisfy that `.X impls J`.
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE+11]]:1: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: final impl forall [T: type] T as GivesI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE-14]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface I { let X: type; }
// CHECK:STDERR:                   ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE+4]]:1: error: constraint `GivesI` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as GivesI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as GivesI {}

// --- fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
  require impls I where .X impls J;
}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE+11]]:1: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: final impl forall [T: type] T as GivesI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE-12]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface I { let X: type; }
// CHECK:STDERR:                   ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE+4]]:1: error: constraint `GivesI` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as GivesI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as GivesI {}

// --- rewrite_in_facet_type_before_with_generic_param.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J(T: type) {}

impl () as J(()) {}

// This uses the earlier rewrite to see that we need `() impls J(())`.
final impl forall [T: type] T as I where .X = () and .X impls J(.X) {}

// --- fail_todo_rewrite_in_facet_type_after_with_generic_param.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J(T: type) {}

impl () as J(()) {}

// We don't see the rewrite until later, but when we look at the constraints
// required, we still need to see that we need `() impls J(())`.
// CHECK:STDERR: fail_todo_rewrite_in_facet_type_after_with_generic_param.carbon:[[@LINE+4]]:1: error: constraint `I where T.(I.X) impls J(T.(I.X)) and .(I.X) = ()` being implemented requires that `T.(I.X)` implements `J(T.(I.X))` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: final impl forall [T: type] T as I where .X impls J(.X) and .X = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T: type] T as I where .X impls J(.X) and .X = () {}
