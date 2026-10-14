// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/require_satisified_in_named_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/require_satisified_in_named_constraint.carbon

// --- fail_missing_require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// () does not need to impl Z yet.
impl () as N;

// () needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_concrete_type.carbon:[[@LINE+4]]:1: error: constraint `N` being implemented requires that `()` implements `Z` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: impl () as N {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as N {}

// --- fail_missing_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// T does not need to impl Z yet.
impl forall [T: type] T as N;

// T needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_symbolic_type.carbon:[[@LINE+4]]:1: error: constraint `N` being implemented requires that `T` implements `Z` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR: impl forall [T: type] T as N {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T: type] T as N {}

// --- require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// () does not need to impl Z yet.
impl () as N;

// Now we know () impls Z.
impl () as Z;

// So no error here.
impl () as N {}

impl () as Z {}

// --- require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// T does not need to impl Z yet.
impl forall [T: type] T as N;

// Now we know T impls Z.
impl forall [T: type] T as Z;

// So no error here.
impl forall [T: type] T as N {}

impl forall [T: type] T as Z {}

// --- require_for_symbolic_type_impl_matches_same_interface.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// This only matches T that impl Z so no error here.
impl forall [T: Z] T as N {}

// --- require_for_generic_interfaces.carbon
library "[[@TEST_NAME]]";

interface Y(T: type) {}
interface Z(T: type) {}

class A;
class B;

constraint N(U: type) {
  extend require impls Y(U);
  require impls Z(B);
}

// () does not need to impl Z(B) yet.
impl () as N(A);

// Now we know () impls Z(B).
impl () as Z(B);

// So no error here.
impl () as N(A) {}

impl () as Z(B) {}

// --- require_non_extended_rewrite.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z { let Z1: type; }

constraint N {
  extend require impls Y;
  require impls Z where .Z1 = ();
}

impl () as N;

// Match the requirement.
impl () as Z where .Z1 = ();

impl () as N {}

impl () as Z where .Z1 = () {}

// --- fail_require_non_extended_rewrite_wrong_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z { let Z1: type; }

constraint N {
  extend require impls Y;
  require impls Z where .Z1 = ();
}

impl () as N;

// Wrong type for the requirement.
impl () as Z where .Z1 = {};

// CHECK:STDERR: fail_require_non_extended_rewrite_wrong_type.carbon:[[@LINE+8]]:1: error: constraint `N` being implemented requires that `().(Z.Z1)` equals `()`; found `{}` and `()` [IdentifiedRequireImplsNotEquivalent]
// CHECK:STDERR: impl () as N {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_require_non_extended_rewrite_wrong_type.carbon:[[@LINE-6]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as Z where .Z1 = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as N {}

impl () as Z where .Z1 = () {}

// --- require_non_extended_rewrite_same_interface.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }

constraint N {
  // Extend Z without a rewrite.
  extend require impls Z;
  // But require `.Z1` is () anyway.
  require impls Z where .Z1 = ();
}

impl () as N where .Z1 = () {}

// --- fail_require_non_extended_rewrite_same_interface_wrong_type.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }

constraint N {
  // Extend Z without a rewrite.
  extend require impls Z;
  // But require .Z1 is () anyway.
  require impls Z where .Z1 = ();
}

// Wrong type for `.Z1`.
// CHECK:STDERR: fail_require_non_extended_rewrite_same_interface_wrong_type.carbon:[[@LINE+4]]:1: error: constraint `N where .(Z.Z1) = {}` being implemented requires that `().(Z.Z1)` equals `()`; found `{}` and `()` [IdentifiedRequireImplsNotEquivalent]
// CHECK:STDERR: impl () as N where .Z1 = {} {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as N where .Z1 = {} {}
