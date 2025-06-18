// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/facet_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/facet_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

fn F(T:! I where .T = {}) {}

// --- success_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; let U:! type; }

fn F(T:! I where .T = .U) {}

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+4]]:10: error: associated constant `.(L.W)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! L where .W = {} and .W = ()) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .W = {} and .W = ()) {}

// --- fail_two_different_first_associated.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; let X:! type; }

// CHECK:STDERR: fail_two_different_first_associated.carbon:[[@LINE+4]]:10: error: associated constant `.(L.W)` given two different values `.(L.X)` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! L where .W = .X and .W = ()) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .W = .X and .W = ()) {}

// --- fail_two_different_second_associated.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; let X:! type; }

// CHECK:STDERR: fail_two_different_second_associated.carbon:[[@LINE+4]]:10: error: associated constant `.(L.W)` given two different values `()` and `.(L.X)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! L where .W = () and .W = .X) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .W = () and .W = .X) {}

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+4]]:23: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! L where .W = BAD5 and .W = ()) {}
// CHECK:STDERR:                       ^~~~
// CHECK:STDERR:
fn F(T:! L where .W = BAD5 and .W = ()) {}

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:35: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! L where .W = {} and .W = BAD6) {}
// CHECK:STDERR:                                   ^~~~
// CHECK:STDERR:
fn F(T:! L where .W = {} and .W = BAD6) {}

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+8]]:23: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! L where .W = BAD7 and .W = BAD8) {}
// CHECK:STDERR:                       ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+4]]:37: error: name `BAD8` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! L where .W = BAD7 and .W = BAD8) {}
// CHECK:STDERR:                                     ^~~~
// CHECK:STDERR:
fn F(T:! L where .W = BAD7 and .W = BAD8) {}

// --- fail_two_different_combined_from_bitand.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different_combined_from_bitand.carbon:[[@LINE+4]]:10: error: associated constant `.(L.W)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! (L where .W = {}) & (L where .W = ())) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! (L where .W = {}) & (L where .W = ())) {}

// --- two_different_combined_from_impl_and_facet.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
impl forall [T:! M] T as L where .W = () {}

fn F(T:! M & (L where .W = {})) {}

class C;
impl C as L where .W = {} {}
impl C as M {}

fn G() {
  F(C);
}

// --- fail_two_different_combined_from_final_impl_and_facet.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

fn G(T:! M & L, a: T.W) -> () { return a; }

fn H(T:! L where .W = {}, a: T.W) -> {} { return a; }

fn F(T:! M & (L where .W = {}), a: T.W) {
  // TODO: One of `b` or `c` must fail, because `T.W` is either found to be `()` from
  // the impl or `{}` from the facet type of T.

  let b: () = G(T, a);
  // CHECK:STDERR: fail_two_different_combined_from_final_impl_and_facet.carbon:[[@LINE+10]]:15: error: cannot implicitly convert expression of type `()` to `{}` [ConversionFailure]
  // CHECK:STDERR:   let c: {} = H(T, a);
  // CHECK:STDERR:               ^~~~~~~
  // CHECK:STDERR: fail_two_different_combined_from_final_impl_and_facet.carbon:[[@LINE+7]]:15: note: type `()` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c: {} = H(T, a);
  // CHECK:STDERR:               ^~~~~~~
  // CHECK:STDERR: fail_two_different_combined_from_final_impl_and_facet.carbon:[[@LINE-13]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn H(T:! L where .W = {}, a: T.W) -> {} { return a; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let c: {} = H(T, a);
}

// --- fail_many_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_many_different.carbon:[[@LINE+4]]:10: error: associated constant `.(L.W)` given two different values `((), (), ())` and `({}, (), ())` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn G(T:! L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {}, ()) and .W = ({}, (), {})) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(T:! L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {}, ()) and .W = ({}, (), {})) {}

// --- rewrite_uses_second_facet.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

fn F(T:! M where .X = (), U:! M where .Y = T.X) -> U.Y {
  return ();
}

// --- fail_rewrite_conflicts_with_second_facet.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

// CHECK:STDERR: fail_rewrite_conflicts_with_second_facet.carbon:[[@LINE+4]]:31: error: associated constant `.(M.Y)` given two different values `T.(M.X)` and `.(M.X)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! M where .X = (), U:! M where .Y = T.X and .Y = .X) {}
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = (), U:! M where .Y = T.X and .Y = .X) {}

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

fn F(T:! M where .X = {} and .X = {}) {}

fn G(T:! M where .X = {}) {
  F(T);
}

// --- repeated_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

fn F(T:! M where .X = .Y and .X = .Y) {}

fn G(T:! M where .X = () and .Y = ()) {
  F(T);
}

// --- repeated_with_bitand.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

fn F1(T:! (M where .X = .Y) & (M where .X = .Y and .Y = ())) -> T.X {
  return ();
}

fn F2(T:! (M where .X = .Y and .Y = ()) & (M where .X = .Y)) -> T.X {
  return ();
}

// --- fail_repeated_and_different.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// CHECK:STDERR: fail_repeated_and_different.carbon:[[@LINE+4]]:10: error: associated constant `.(M.X)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! M where .X = {} and .X = () and .X = {}) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = {} and .X = () and .X = {}) {}

// --- fail_cycle_single.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// CHECK:STDERR: fail_cycle_single.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! M where .X = .X) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = .X) {}

// --- fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; let Z:! type; }

// This fails because it resolves to `.X = .X` which is cyclical.
//
// CHECK:STDERR: fail_cycle.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! M where .X = .Y and .Y = .X and .Z = ()) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = .Y and .Y = .X and .Z = ()) {}

// --- fail_cycle_between_interfaces.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}
interface J {
  let X3:! type;
}

// This fails because it resolves to `.X1 = .X1` which is cyclical.
//
// CHECK:STDERR: fail_cycle_between_interfaces.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(I.X2)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn G(T:! I & J where .X1 = .X3 and .X2 = .X1 and .X3 = .X2) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(T:! I & J where .X1 = .X3 and .X2 = .X1 and .X3 = .X2) {}

// --- fail_indirect_cycle.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

// This fails because it resolves to `.X1 = .X1**` which is cyclical.
//
// CHECK:STDERR: fail_indirect_cycle.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(I.X1)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! I where .X1 = .X2* and .X2 = .X1*);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! I where .X1 = .X2* and .X2 = .X1*);

class C(T:! type);
// This fails because it resolves to `.X1 = C(C(.X1))` which is cyclical.
//
// CHECK:STDERR: fail_indirect_cycle.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(I.X1)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn G(T:! I where .X1 = C(.X2) and .X2 = C(.X1));
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(T:! I where .X1 = C(.X2) and .X2 = C(.X1));

// --- fail_complex_indirect_cycle.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
}

class C(T:! type, U:! type);

// This fails because it resolves to `.X1 = C(C(.X3, .X1), .X3)` which is
// cyclical.
//
// CHECK:STDERR: fail_complex_indirect_cycle.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(I.X1)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! I where .X1 = C(.X2, .X3) and .X2 = C(.X3, .X1));
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! I where .X1 = C(.X2, .X3) and .X2 = C(.X3, .X1));

// --- fail_cycle_and_conflict.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

// CHECK:STDERR: fail_cycle_and_conflict.carbon:[[@LINE+8]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! M where .X = .X and .Y = () and .Y = {}) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_cycle_and_conflict.carbon:[[@LINE+4]]:10: error: associated constant `.(M.Y)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! M where .X = .X and .Y = () and .Y = {}) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = .X and .Y = () and .Y = {}) {}

// --- non-type.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

fn F(T:! N where .Y = {.a = {}}) { }

// --- non-type_repeated.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

fn F(T:! N where .Y = {.a = {}} and .Y = {.a = {}}) { }

// --- fail_non-type_different.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: type};
}

// CHECK:STDERR: fail_non-type_different.carbon:[[@LINE+4]]:10: error: associated constant `.(N.Y)` given two different values `{.a = {}}` and `{.a = ()}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! N where .Y = {.a = {}} and .Y = {.a = ()}) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! N where .Y = {.a = {}} and .Y = {.a = ()}) {}

// --- fail_todo_cycle_through_self_reference.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  // TODO: This should not be an error.
  //
  // CHECK:STDERR: fail_todo_cycle_through_self_reference.carbon:[[@LINE+7]]:11: error: associated constant has incomplete type `Z` [IncompleteTypeInAssociatedConstantDecl]
  // CHECK:STDERR:   let U:! Z;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_todo_cycle_through_self_reference.carbon:[[@LINE-7]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let U:! Z;
}

// TODO: Should be diagnosed as a cycle.
fn F(A:! Z where .T = .U.T and .U = .Self) {}

// --- fail_todo_reference_same_constant_in_different_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  // TODO: This should not be an error.
  //
  // CHECK:STDERR: fail_todo_reference_same_constant_in_different_self.carbon:[[@LINE+7]]:11: error: associated constant has incomplete type `Z` [IncompleteTypeInAssociatedConstantDecl]
  // CHECK:STDERR:   let U:! Z;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_todo_reference_same_constant_in_different_self.carbon:[[@LINE-7]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let U:! Z;
}

// TODO: Should not be diagnosed as a cycle, once the incorrect failure above is
// fixed.
fn F(A:! Z where .T = (), B:! Z where .T = .U.T and .U = A) {}

// --- fail_todo_non_cycle_with_self_reference.carbon
library "[[@TEST_NAME]]";

interface Z {
  // TODO: This should not be an error.
  //
  // CHECK:STDERR: fail_todo_non_cycle_with_self_reference.carbon:[[@LINE+7]]:11: error: associated constant has incomplete type `Z` [IncompleteTypeInAssociatedConstantDecl]
  // CHECK:STDERR:   let T:! Z;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_todo_non_cycle_with_self_reference.carbon:[[@LINE-6]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let T:! Z;
  let U:! type;
  // TODO: This should not be an error.
  //
  // CHECK:STDERR: fail_todo_non_cycle_with_self_reference.carbon:[[@LINE+7]]:11: error: associated constant has incomplete type `Z` [IncompleteTypeInAssociatedConstantDecl]
  // CHECK:STDERR:   let V:! Z;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_todo_non_cycle_with_self_reference.carbon:[[@LINE-17]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let V:! Z;
}

// TODO: Should not be diagnosed as a cycle, once the incorrect failure above is
// fixed.
fn F(A:! Z where .T = .V.U and .V = .Self and .U = ()) -> A.T {
  return ();
}
