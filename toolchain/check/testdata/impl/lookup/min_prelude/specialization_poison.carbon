// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization_poison.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization_poison.carbon

// --- todo_fail_final_poisoned_concrete_query.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  fn F[self: Self]() -> T;
}

impl forall [U:! type] U as I where .T = () {
  fn F[self: Self]() -> () { return (); }
}

fn H[W:! type](v: W) -> W.(I.T) {
  return v.(I.F)();
}

class C { adapt (); }

fn G(p: C*) -> () {
  // This concrete impl lookup query poisons any further specializations.
  return H(p);
}

// TODO: Diagnose this as a poisoned specialization.
impl C* as I where .T = C {
  fn F[self: Self]() -> C { return *self; }
}

// --- todo_fail_final_poisoned_concrete_query_nested_type_in_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  fn F[self: Self]() -> T;
}

class C(U:! type) {}

impl forall [U:! type] C(U) as I where .T = () {
  fn F[self: Self]() -> () { return (); }
}

fn G(c: C(())) -> () {
  // This concrete impl lookup query poisons any further specializations.
  return c.(I.F)();
}

// TODO: Diagnose this as a poisoned specialization.
impl C(()) as I where .T = {} {
  fn F[self: Self]() -> {} { return {}; }
}

// --- todo_fail_final_poisoned_concrete_query_nested_type_in_interface.carbon
library "[[@TEST_NAME]]";

interface I(U:! type) {
  let T:! type;
  fn F[self: Self]() -> T;
}

impl forall [U:! type] U as I(U) where .T = () {
  fn F[self: Self]() -> () { return (); }
}

class C {}

fn G(c: C) -> () {
  // This concrete impl lookup query poisons any further specializations.
  return c.(I(C).F)();
}

// TODO: Diagnose this as a poisoned specialization.
impl C as I(C) where .T = {} {
  fn F[self: Self]() -> {} { return {}; }
}

// --- todo_fail_final_poisoned_by_generic_query.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  fn F[self: Self]() -> T;
}

impl forall [U:! type] U as I where .T = () {
  fn F[self: Self]() -> () { return (); }
}

fn H[W:! type](v: W) -> W.(I.T) {
  return v.(I.F)();
}

// This function could return a concrete `X` if it saw the `final` impl below.
fn G[X:! type](p: X*) -> (X*).(I.T) {
  return H(p);
}

// TODO: This should be diagnosed as poisoned, as H() uses an associated
// constant from the `impl` which was treated as a symbolic, but this would
// change it to be a concrete type.
final impl forall [V:! type] V* as I where .T = V {
  fn F[self: Self]() -> V { return *self; }
}
