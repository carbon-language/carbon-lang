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

// --- repeated_concrete_value_and_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

fn F1(T:! M where .X = () and .Y = .X and .X = .Y) {}

fn F2(T:! M where .X = () and .X = .X) {}

fn G(T:! M where .X = () and .Y = ()) {
  F1(T);
  F2(T);
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

// This fails because it resolves to `.X = .X` which is cyclical.
//
// CHECK:STDERR: fail_cycle_single.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! M where .X = .X) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = .X) {}

// Even though `.X = ()` is specified, the rewrites are resolved left to right
// and a cycle `.X = .X` is found first.
//
// CHECK:STDERR: fail_cycle_single.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn G(T:! M where .X = .X and .X = ()) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(T:! M where .X = .X and .X = ()) {}

// --- fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; let Z:! type; }

// This fails because it resolves to `.X = .X` which is cyclical.
// The value of .X and .Y becomes <error> but .Z is still valid.
//
//@dump-sem-ir-begin
// CHECK:STDERR: fail_cycle.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! M where .X = .Y and .Y = .X and .Z = ()) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = .Y and .Y = .X and .Z = ()) {}
//@dump-sem-ir-end

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
// CHECK:STDERR: fail_cycle_between_interfaces.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(I.X1)` [FacetTypeConstraintCycle]
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

// --- exponential_large.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
  let T4:! type;
  let T5:! type;
  let T6:! type;
  let T7:! type;
  let T8:! type;
  let T9:! type;
}

// A naive attempt to resolve the rewrite rules will run take minutes to
// complete, since the resulting RHS values are exponential in size, and a naive
// approach can recursively rebuild the RHS values from the ground up
// repeatedly.
fn F(
    T:! Z where
      .T0 = (.T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1) and
      .T1 = (.T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2) and
      .T2 = (.T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3) and
      .T3 = (.T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4) and
      .T4 = (.T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5) and
      .T5 = (.T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6) and
      .T6 = (.T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7) and
      .T7 = (.T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8) and
      .T8 = (.T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9) and
      .T9 = ()
    );

// --- fail_exponential_large_cycle.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
  let T4:! type;
  let T5:! type;
  let T6:! type;
  let T7:! type;
  let T8:! type;
  let T9:! type;
}

// A naive attempt to resolve the rewrite rules will run take minutes to
// complete, since the resulting RHS values are exponential in size, and a naive
// approach can recursively rebuild the RHS values from the ground up
// repeatedly.
fn F(
    // CHECK:STDERR: fail_exponential_large_cycle.carbon:[[@LINE+4]]:9: error: found cycle in facet type constraint for `.(Z.T9)` [FacetTypeConstraintCycle]
    // CHECK:STDERR:     T:! Z where
    // CHECK:STDERR:         ^~~~~~~
    // CHECK:STDERR:
    T:! Z where
      .T9 = .T0 and
      .T8 = (.T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9, .T9) and
      .T7 = (.T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8, .T8) and
      .T6 = (.T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7, .T7) and
      .T5 = (.T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6, .T6) and
      .T4 = (.T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5, .T5) and
      .T3 = (.T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4, .T4) and
      .T2 = (.T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3, .T3) and
      .T1 = (.T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2, .T2) and
      .T0 = (.T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1, .T1)
    );

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

// --- fail_cycle_with_unrelated_associated_constant.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
}

// CHECK:STDERR: fail_cycle_with_unrelated_associated_constant.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(Z.T0)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! Z where .T0 = .T1 and .T1 = .T0 and .T2 = .T3) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! Z where .T0 = .T1 and .T1 = .T0 and .T2 = .T3) {}

// --- fail_cycle_with_branching_in_rhs.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
  let T4:! type;
}

// TODO: There should only be one diagnostic here.
//
// CHECK:STDERR: fail_cycle_with_branching_in_rhs.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(Z.T1)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn F(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T3 = .T1) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T3 = .T1) {}

// CHECK:STDERR: fail_cycle_with_branching_in_rhs.carbon:[[@LINE+4]]:10: error: found cycle in facet type constraint for `.(Z.T0)` [FacetTypeConstraintCycle]
// CHECK:STDERR: fn G(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T3 = .T0) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T3 = .T0) {}

// --- no_cycle_with_branching_in_rhs.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T0:! type;
  let T1:! type;
  let T2:! type;
  let T3:! type;
  let T4:! type;
  let T5:! type;
}

// These create misdiagnostics if the resolving algorithms messes up tracking
// its stack during replacements by leaving either of .T2 or .T3 on the stack
// (from the RHS of .T1) while resolving the other. Or it can fail to apply the
// () up the chain correctly.

fn F(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T4 = () and .T3 = .T2) -> T.T0 {
  return ((), ());
}

fn G(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T4 = .T5 and .T5 = () and .T3 = .T2) -> T.T0 {
  return ((), ());
}

fn H(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = (.T4, ()) and .T3 = .T2 and .T4 = {}) -> T.T0 {
  return (({}, ()), ({}, ()));
}

fn I(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T3 = .T2 and .T4 = ()) -> T.T0 {
  return ((), ());
}

fn J(T:! Z where .T0 = .T1 and .T1 = (.T2, .T3) and .T2 = .T4 and .T4 = .T5 and .T3 = .T2 and .T5 = ()) -> T.T0 {
  return ((), ());
}

// CHECK:STDOUT: --- fail_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%Y [concrete]
// CHECK:STDOUT:   %assoc2: %M.assoc_type = assoc_entity element2, @M.%Z [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %M.lookup_impl_witness, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem2: type = impl_witness_access %M.lookup_impl_witness, element2 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_12.1: type = splice_block %.loc13_12.2 [concrete = <error>] {
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc13_18: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc13_18: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc13_18: type = facet_access_type %.Self.ref.loc13_18 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_18: type = converted %.Self.ref.loc13_18, %.Self.as_type.loc13_18 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc13_18: type = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.Self.ref.loc13_23: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref.loc13_23: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc13_23: type = facet_access_type %.Self.ref.loc13_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_23: type = converted %.Self.ref.loc13_23, %.Self.as_type.loc13_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem1.loc13_23: type = impl_witness_access constants.%M.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %.Self.ref.loc13_30: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref.loc13_30: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc13_30: type = facet_access_type %.Self.ref.loc13_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_30: type = converted %.Self.ref.loc13_30, %.Self.as_type.loc13_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem1.loc13_30: type = impl_witness_access constants.%M.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %.Self.ref.loc13_35: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc13_35: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc13_35: type = facet_access_type %.Self.ref.loc13_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_35: type = converted %.Self.ref.loc13_35, %.Self.as_type.loc13_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc13_35: type = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.Self.ref.loc13_42: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:       %.Self.as_type.loc13_42: type = facet_access_type %.Self.ref.loc13_42 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_42: type = converted %.Self.ref.loc13_42, %.Self.as_type.loc13_42 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem2: type = impl_witness_access constants.%M.lookup_impl_witness, element2 [symbolic_self = constants.%impl.elem2]
// CHECK:STDOUT:       %.loc13_48.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc13_48.2: type = converted %.loc13_48.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc13_12.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc13_18, %impl.elem1.loc13_23
// CHECK:STDOUT:         requirement_rewrite %impl.elem1.loc13_30, %impl.elem0.loc13_35
// CHECK:STDOUT:         requirement_rewrite %impl.elem2, %.loc13_48.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {}
// CHECK:STDOUT:
