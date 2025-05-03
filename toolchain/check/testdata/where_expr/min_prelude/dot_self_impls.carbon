// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/min_prelude/dot_self_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/min_prelude/dot_self_impls.carbon

// --- compound_member_access_through_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // Can access members of `I` using either kind of member access.
  T.FNonInstance();
  T.(I.FNonInstance)();

  x.FNonInstance();
  x.FSelf();
  x.(I.FSelf)();

  // Can still find members of `J` using compound member access,
  // even though they are not available via `T.GNonInstance` or
  // `x.GSelf`.
  T.(J.GNonInstance)();
  x.(J.GSelf)();
}

fn TypeNotJ[T:! type where .Self impls J](x: T) {
  T.(J.GNonInstance)();
  x.(J.GSelf)();
}

// --- no_name_conflict_with_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F() -> ();
}

interface J {
  fn F() -> {};
}

fn INotJ(T:! I where .Self impls J) {
  // Gets `T.(I.F)`. Doesn't consider `T.(J.F)`, so no ambiguity.
  let x: () = T.F();
}

// --- fail_name_lookup_through_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_through_where_self_impls.carbon:[[@LINE+4]]:3: error: member name `G` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- fail_name_lookup_with_type.carbon
library "[[@TEST_NAME]]";

interface J {
  fn G();
}

fn TypeNotJ(T:! type where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_with_type.carbon:[[@LINE+4]]:3: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- fail_facet_type_simple_member_access.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B { fn Bfn(); }

fn F(T:! A & B) {
  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:6: error: member name `Bfn` not found in `A` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.((A where .Self impls B).Bfn)();
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.((A where .Self impls B).Bfn)();

  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:3: error: member name `Bfn` not found in `A` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   (A where .Self impls B).Bfn;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (A where .Self impls B).Bfn;
}

// --- fail_name_lookup_instance.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GNonInstance` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GNonInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  x.GNonInstance();

  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GSelf` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GSelf();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.GSelf();
}

// --- compare_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I;
interface J;
interface K;

fn Test() {
  AssertSame(Type(I), Type(I where .Self impls I));
  AssertSame(Type(I), Type(I where .Self impls I & I));
  AssertSame(Type(I & J), Type(J & I where .Self impls I));
  AssertSame(Type(I & J), Type(J & I where .Self impls J));
  AssertSame(Type(I & J), Type(J & I where .Self impls I & J));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls I & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls J & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K), Type(I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (J where .Self impls K)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (K where .Self impls J)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (type where .Self impls (J where .Self impls K))));
}

interface I {}
interface J {}
interface K {}

// --- compare_equal_with_associated_constant.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I { let A:! type; }
interface J { let A:! type; }

fn Test() {
  AssertSame(Type((I where .A = ()) & J),
             Type(I & J where .Self impls (I where .A = ())));
  AssertSame(Type(I & (J where .A = {})),
             Type(I & J where .Self impls (J where .A = {})));

  AssertSame(Type((I where .A = ()) & (J where .A = {})),
             Type(I & J where .Self impls (I where .A = ())
                    and .Self impls (J where .A = {})));
  AssertSame(Type((I where .A = ()) & (J where .A = {})),
             Type(I & J where .Self impls (I where .A = ()) & (J where .A = {})));
}

// --- fail_compare_not_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I {}
interface J {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   Same(Type(I where .Self impls J), Type(J where .Self impls I));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Same(Type(I where .Self impls J), Type(J where .Self impls I));
}

// --- impl_as.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface J { let T:! type; }

// Interfaces to the right of the `where` don't interfere with the
// "can only impl a facet type with a single interface" rule. Only
// the interface being implemented needs to have all of its
// associated constants set
impl {.a: C} as I where .Self impls J {}
impl {.b: C} as J where .Self impls I and .T = () {}

// Rewrite constraints can appear inside the `where .Self impls`.
impl {.c: C} as J where .Self impls (I & J where .T = ()) {}

// --- impl_with_rewrite_of_interface_not_being_implemented.carbon
library "[[@TEST_NAME]]";

interface I { let A:! type; }
interface J { let A:! type; }

class C {}

// Implementation of `C as J`.
impl C as J where .A = {} {}

// This is an implementation of `I` with `I.A = ()`. The requirement
// that `C` also impls `J where .A = {}` is an additional constraint
// that must be satisfied (and is satisfied by the impl above, though
// that currently isn't checked), but doesn't affect `C as I`.
impl C as I where .Self impls
    (J where .A = {} and .Self impls (I where .A = ())) {}

let x: C.(I.A) = ();
let y: C.(J.A) = {};
