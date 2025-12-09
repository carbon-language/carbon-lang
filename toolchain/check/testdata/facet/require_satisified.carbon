// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_satisified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_satisified.carbon

// --- todo_fail_missing_require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// TODO: This should be an error. () needs to impl Z at the start of the
// definition.
impl () as Y {}

// --- todo_fail_missing_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// TODO: This should be an error. T needs to impl Z at the start of the
// definition.
impl forall [T:! type] T as Y {}

// --- require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// Now we know () impls Z.
impl () as Z;

// So no error here.
impl () as Y {}

impl () as Z {}

// --- require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// Now we know T impls Z.
impl forall [T:! type] T as Z;

// So no error here.
impl forall [T:! type] T as Y {}

impl forall [T:! type] T as Z {}

// --- require_for_symbolic_type_impl_matches_same_interface.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// This only matches T that impl Z so no error here.
impl forall [T:! Z] T as Y {}
