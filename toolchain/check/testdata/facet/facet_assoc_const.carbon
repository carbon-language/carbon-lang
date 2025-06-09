// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
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

// --- fail_todo_two_different_combined_from_final_impl_and_facet.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

fn G(T:! M & L, a: T.W) -> () { return a; }

// TODO: This should not fail.
//
// CHECK:STDERR: fail_todo_two_different_combined_from_final_impl_and_facet.carbon:[[@LINE+7]]:43: error: cannot implicitly convert expression of type `T.(L.W)` to `{}` [ConversionFailure]
// CHECK:STDERR: fn H(T:! L where .W = {}, a: T.W) -> {} { return a; }
// CHECK:STDERR:                                           ^~~~~~~~~
// CHECK:STDERR: fail_todo_two_different_combined_from_final_impl_and_facet.carbon:[[@LINE+4]]:43: note: type `T.(L.W)` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn H(T:! L where .W = {}, a: T.W) -> {} { return a; }
// CHECK:STDERR:                                           ^~~~~~~~~
// CHECK:STDERR:
fn H(T:! L where .W = {}, a: T.W) -> {} { return a; }

fn F(T:! M & (L where .W = {}), a: T.W) {
  // TODO: One of `b` or `c` must fail, because `T.W` is either found to be `()` from
  // the impl or `{}` from the facet type of T.

  let b: () = G(T, a);
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

// --- fail_repeated_and_different.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// CHECK:STDERR: fail_repeated_and_different.carbon:[[@LINE+4]]:10: error: associated constant `.(M.X)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: fn F(T:! M where .X = {} and .X = () and .X = {}) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! M where .X = {} and .X = () and .X = {}) {}

// --- todo_fail_cycle_single.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// TODO: This should fail, as it resolves to `.X = .X` which is cyclical.
fn F(T:! M where .X = .X) {}

// --- todo_fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

// TODO: This should fail, as it resolves to `.X = .X` which is cyclical.
fn F(T:! M where .X = .Y and .Y = .X) {}

// --- todo_fail_cycle_between_interfaces.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}
interface J {
  let X3:! type;
}

// TODO: This should fail, as it resolves to `.X1 = .X1` which is cyclical.
fn G(T:! I & J where .X1 = .X3 and .X2 = .X1 and .X3 = .X2) {}

// --- todo_fail_indirect_cycle.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}

// TODO: This should fail, as it resolves to `.X1 = .X1**` which is cyclical.
fn F(T:! I where .X1 = .X2* and .X2 = .X1*);

class C(T:! type);
// TODO: This should fail, as it resolves to `.X1 = C(C(.X1))` which is
// cyclical.
fn G(T:! I where .X1 = C(.X2) and .X2 = C(.X1));

// --- todo_fail_complex_indirect_cycle.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
  let X3:! type;
}

class C(T:! type, U:! type);

// TODO: This should fail, as it resolves to `.X1 = C(C(.X3, .X1), .X3)` which
// is cyclical.
fn F(T:! I where .X1 = C(.X2, .X3) and .X2 = C(.X3, .X1));

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
