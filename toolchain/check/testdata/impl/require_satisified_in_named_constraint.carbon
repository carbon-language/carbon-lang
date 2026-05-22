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

// --- todo_fail_missing_require_for_concrete_type.carbon
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
// TODO: RequireImplsNotImplemented error that () does not implement Z.
impl () as N {}

// --- todo_fail_missing_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as N;

// T needs to impl Z at the start of the definition.
// TODO: RequireImplsNotImplemented error that () does not implement Z.
impl forall [T:! type] T as N {}

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
impl forall [T:! type] T as N;

// Now we know T impls Z.
impl forall [T:! type] T as Z;

// So no error here.
impl forall [T:! type] T as N {}

impl forall [T:! type] T as Z {}

// --- require_for_symbolic_type_impl_matches_same_interface.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {}

constraint N {
  extend require impls Y;
  require impls Z;
}

// This only matches T that impl Z so no error here.
impl forall [T:! Z] T as N {}

// --- require_for_generic_interfaces.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {}
interface Z(T:! type) {}

class A;
class B;

constraint N(U:! type) {
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
