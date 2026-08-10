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
impl forall [T: type] T as I where .X = () and .X impls J {}

// --- fail_rewrite_in_facet_type_before_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_facet_type_before_missing_impl.carbon:[[@LINE+2]]:1-59: error: constraint `I where () impls J and .(I.X) = ()` being implemented requires that `()` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl forall [T: type] T as I where .X = () and .X impls J {}

// --- rewrite_in_facet_type_after.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

impl () as J {}

// We don't see the rewrite until later, but when we look at the constraints
// required, we still need to see that we need `() impls J`.
impl forall [T: type] T as I where .X impls J and .X = () {}

// --- fail_rewrite_in_facet_type_after_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_facet_type_after_missing_impl.carbon:[[@LINE+2]]:1-59: error: constraint `I where .(I.X) impls J and .(I.X) = ()` being implemented requires that `()` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl forall [T: type] T as I where .X impls J and .X = () {}

// --- fail_todo_rewrite_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
}

impl () as J {}

// The rewrite from `extend` is inherited and should satisfy that `.X impls J`.
// CHECK:STDERR: fail_todo_rewrite_in_named_constraint.carbon:[[@LINE+3]]:1-52: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_todo_rewrite_in_named_constraint.carbon:[[@LINE-11]]:19-25: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
impl forall [T: type] T as GivesI where .X impls J {}

// --- fail_rewrite_in_named_constraint_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_in_named_constraint_missing_impl.carbon:[[@LINE+3]]:1-52: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_rewrite_in_named_constraint_missing_impl.carbon:[[@LINE-9]]:19-25: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
impl forall [T: type] T as GivesI where .X impls J {}

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
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE+5]]:1-35: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE-12]]:19-25: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_rewrite_and_impls_in_named_constraint.carbon:[[@LINE+2]]:1-35: error: constraint `GivesI` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl forall [T: type] T as GivesI {}

// --- fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

constraint GivesI {
  extend require impls I where .X = ();
  require impls I where .X impls J;
}

// .X does not impl J.
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE+5]]:1-35: error: associated constant X not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE-10]]:19-25: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_rewrite_and_impls_in_named_constraint_missing_impl.carbon:[[@LINE+2]]:1-35: error: constraint `GivesI` being implemented requires that `T.(I.X)` implements `J` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl forall [T: type] T as GivesI {}

// --- rewrite_in_facet_type_before_with_generic_param.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J(T: type) {}

impl () as J(()) {}

// This uses the earlier rewrite to see that we need `() impls J(())`.
impl forall [T: type] T as I where .X = () and .X impls J(.X) {}

// --- rewrite_in_facet_type_after_with_generic_param.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J(T: type) {}

impl () as J(()) {}

// We don't see the rewrite until later, but when we look at the constraints
// required, we still need to see that we need `() impls J(())`.
impl forall [T: type] T as I where .X impls J(.X) and .X = () {}

// --- impl_lookups_in_definition_find_own_impl.carbon
library "[[@TEST_NAME]]";

class C {}
class G(T: type) {}

interface Z(R: type) {
  let Result: type;
}
interface Y(S: type) {}

impl forall [T: type] () as Y(G(T)) {}
impl forall [T: type] {} as Y(G(T)) {}

// This is a blanket impl that overlaps with (but is less specific than)
// `G(C) as Z(G(C))`, which is the next impl below.
impl forall [T: type, U: type] G(T) as Z(U) where .Result impls Y(G(T)) and .Result = {} {}

// The query `G(C) as Z(G(C))` is concrete so any lookups will result in a final
// witness. The constraint facet type contains `.Result` which causes a lookup
// for `G(C) as Z(G(C))`. But this lookup occurs before the impl declaration is
// even completed, so the impl does not yet exist according to check.
//
// The naive result is that the lookup finds overlapping impl above, and
// `.Result` evaluates to `{}` which is incorrect. Then a later query finds this
// impl and `.Result` evaluates to `()`. We diagnose this a poisoning error,
// where this impl changes the possible value of `.Result` after a concrete
// query was made for it.
//
// The correct result is for the witness of `G(C) as Z(G(C))` to point to this
// impl, even though it does not yet exist. No impl lookup should occur for the
// interface being impl'd from within the declaration. Then `.Result` consistent
// evalutes to `()`.
impl G(C) as Z(G(C)) where .Result impls Y(G(C)) and .Result = () {}

fn F() {
  G(C) as Z(G(C));
}

// --- impl_lookups_in_definition_find_own_impl_with_named_constraint.carbon
library "[[@TEST_NAME]]";

class C {}
class G(T: type) {}

interface Z(R: type) {
  let Result: type;
}
interface Y(S: type) {}

impl forall [T: type] () as Y(G(T)) {}
impl forall [T: type] {} as Y(G(T)) {}

impl forall [T: type, U: type] G(T) as Z(U) where .Result impls Y(G(T)) and .Result = {} {}

constraint N(T: type) {
  extend require impls Z(G(T));
}

// See impl_lookups_in_definition_find_own_impl.carbon for an explanation of
// this test. We are testing the same things, except we put the impl's target
// interface into a named constraint.
impl G(C) as N(C) where .Result impls Y(G(C)) and .Result = () {}

fn F() {
  G(C) as Z(G(C));
}

// --- lookup_for_impl_with_period_self_in_constraint.carbon
library "[[@TEST_NAME]]";

class C {}
class G(T: type) {}

interface Z(R: type) {
  let Result: type;
}
interface Y(S: type) {}

impl forall [T: type] () as Y(G(T)) {}

// We don't want `.Self` references to escape the `impl` decl, so this tests a
// scenario where they used to escape and cause an infinite cycle.
//
// The impl lookup in `F()` finds this impl, which has a `.Self` in its set of
// constraints (in `.Result impls ...`). The lookup operation can't identify the
// constraint and replace `.Self` without creating a new witness that causes a
// lookup for this impl again.
impl forall [T: type, U: type] G(T) as Z(U) where .Result impls Y(G(T)) and .Result = () {}

fn F() {
  G(C) as Z(G(C));
}

// --- where_that_does_not_constrain_impl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y(T: type) {}

interface Tuple {}
impl () as Tuple {}

// This test puts a `where` expression in two problematic places:
// - The `where` in the generic constraint for the impl should _not_ refer to
//   the impl. It should have a non-final witness for the `.Z1` access.
// - The `where` in the generic argument to `Y` does not constrain the impl's
//   self type so the access `.Z1` should also have a symbolic witness, not a
//   witness for the impl itself.
//
// If done incorrectly, either of the above may become a witness for the impl
// itself, which is incorrect and results in a lookup into an empty witness
// table (for `Y`) when looking for `.Z1`.
//
// Also if done incorrectly, we may declare that lookups in `Z` are for the
// current impl, because of the `Z where ...` in the argument for `Y` if we
// mistake that as being `impl as Z where ...`. That would result in the later
// access `.Self.(Z.Z1)` becoming a witness for the impl itself, which would try
// to find a value for `.Z1` in the empty witness table for `Y`.
impl forall [T: Z where .Z1 = ()] T as Y(Z where .Z1 = ())
  where .Self impls Z and .Self.(Z.Z1) impls Tuple {}
