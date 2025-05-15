// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/facet_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/facet_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

fn F(T:! I where .T = {}) {}

// --- success_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; let U:! type; }

fn F(T:! I where .T = .U) {}

// --- todo_fail_two_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

fn F(T:! L where .W = {} and .W = ()) {}

// --- todo_fail_two_different_first_associated.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; let X:! type; }

fn F(T:! L where .W = .X and .W = ()) {}

// --- todo_fail_two_different_second_associated.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; let X:! type; }

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

// --- todo_fail_two_different_combined_from_bitand.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

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

// --- todo_fail_two_different_combined_from_final_impl_and_facet.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

fn F(T:! M & (L where .W = {})) {}

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

// --- todo_fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

// TODO: This should fail, since there's no canonical form for .X or .Y.
fn F(T:! M where .X = .Y and .Y = .X) {}

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

// --- todo_fail_non-type_different.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: type};
}

fn F(T:! N where .Y = {.a = {}} and .Y = {.a = ()}) {}
