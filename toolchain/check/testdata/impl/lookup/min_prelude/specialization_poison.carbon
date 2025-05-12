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

// --- fail_final_poisoned_concrete_query_in_specific.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  fn F[self: Self]() -> T;
}

impl forall [U:! type] U as I where .T = () {
  fn F[self: Self]() -> () { return (); }
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE+3]]:25: error: found `impl` that would change the result of an earlier use of `C* as I` [PoisonedImplLookupConcreteResult]
// CHECK:STDERR: fn H[W:! type](v: W) -> W.(I.T) {
// CHECK:STDERR:                         ^~~~~~~
fn H[W:! type](v: W) -> W.(I.T) {
  return v.(I.F)();
}

class C { adapt (); }

fn G(p: C*) -> () {
  // This concrete impl lookup query poisons any further specializations.
  return H(p);
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE+7]]:1: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: impl C* as I where .T = C {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE-21]]:1: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR: impl forall [U:! type] U as I where .T = () {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C* as I where .T = C {
  fn F[self: Self]() -> C { return *self; }
}

// --- fail_final_poisoned_concrete_query_in_generic.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = () {}

class C {}

fn H[W:! type](v: W) {
  // This concrete impl lookup query poisons any further specializations.
  // CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE+3]]:10: error: found `impl` that would change the result of an earlier use of `C as I` [PoisonedImplLookupConcreteResult]
  // CHECK:STDERR:   let a: C.(I.T) = ();
  // CHECK:STDERR:          ^~~~~~~
  let a: C.(I.T) = ();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE+7]]:1: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: impl C as I where .T = C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE-15]]:1: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR: impl forall [U:! type] U as I where .T = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as I where .T = C {}

// --- fail_final_poisoned_concrete_query_nested_type_in_self.carbon
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
  // CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE+3]]:10: error: found `impl` that would change the result of an earlier use of `C(()) as I` [PoisonedImplLookupConcreteResult]
  // CHECK:STDERR:   return c.(I.F)();
  // CHECK:STDERR:          ^~~~~~~
  return c.(I.F)();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE+7]]:1: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: impl C(()) as I where .T = {} {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE-15]]:1: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR: impl forall [U:! type] C(U) as I where .T = () {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(()) as I where .T = {} {
  fn F[self: Self]() -> {} { return {}; }
}

// --- fail_final_poisoned_concrete_query_nested_type_in_interface.carbon
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
  // CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE+3]]:10: error: found `impl` that would change the result of an earlier use of `C as I(C)` [PoisonedImplLookupConcreteResult]
  // CHECK:STDERR:   return c.(I(C).F)();
  // CHECK:STDERR:          ^~~~~~~~~~
  return c.(I(C).F)();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE+7]]:1: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: impl C as I(C) where .T = {} {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE-17]]:1: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR: impl forall [U:! type] U as I(U) where .T = () {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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

// --- fail_final_overlaps_earlier_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

// CHECK:STDERR: fail_final_overlaps_earlier_non_final_impl.carbon:[[@LINE+3]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl () as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
impl () as Z(()) {}

// CHECK:STDERR: fail_final_overlaps_earlier_non_final_impl.carbon:[[@LINE+4]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(T) {}

// --- fail_final_overlaps_later_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

final impl forall [T:! type] T as Z(T) {}

// CHECK:STDERR: fail_final_overlaps_later_non_final_impl.carbon:[[@LINE+7]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl () as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_later_non_final_impl.carbon:[[@LINE-5]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Z(()) {}

// --- fail_final_overlaps_earlier_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

class C;

// CHECK:STDERR: fail_final_overlaps_earlier_final_impl.carbon:[[@LINE+3]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: final impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
final impl C as Z(C) {}

// CHECK:STDERR: fail_final_overlaps_earlier_final_impl.carbon:[[@LINE+4]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(T) {}

// --- fail_final_overlaps_later_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

class C;

final impl forall [T:! type] T as Z(T) {}

// CHECK:STDERR: fail_final_overlaps_later_final_impl.carbon:[[@LINE+7]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: final impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_later_final_impl.carbon:[[@LINE-5]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl C as Z(C) {}

// --- fail_non_final_overlaps_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

impl forall [T:! Y] T as Z(T) {}

class C(T:! type);

// No diagnosis here as the type structure is unique.
impl forall [T:! type] T as Z(C(T)) {}

// CHECK:STDERR: fail_non_final_overlaps_non_final_impl.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_non_final_overlaps_non_final_impl.carbon:[[@LINE-10]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T:! Y] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) {}

// --- non_final_concrete_overlaps_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;
class D;

impl C as Z {}
impl D as Z {}

// --- final_partial_overlaps_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

class C;

impl forall [T:! type] T as Z(T) {}

// Partially overlaps the previous impl, which is fine.
final impl C as Z(C) {}

// --- todo_final_partial_overlaps_final_impl_in_self_type.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y {}

class C(T:! type, U:! type);

final impl forall [T:! type] C(T, ()) as Z {}

// TODO: Partially overlaps the previous impl, which may or may not be fine outside of
// a `match_first`, depending how
// https://github.com/carbon-language/carbon-lang/pull/5337 turns out.
final impl forall [U:! type] C((), U) as Z {}

// --- todo_final_partial_overlaps_final_impl_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type, U:! type) {}
interface Y {}

class C;

final impl forall [T:! type] C as Z(T, ()) {}

// TODO: Partially overlaps the previous impl, which may or may not be fine outside of
// a `match_first`, depending how
// https://github.com/carbon-language/carbon-lang/pull/5337 turns out.
final impl forall [U:! type] C as Z((), U) {}
