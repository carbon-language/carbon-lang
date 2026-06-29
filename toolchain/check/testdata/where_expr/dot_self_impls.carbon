// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon

// --- compound_member_access_through_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf(self);
}

interface J {
  fn GNonInstance();
  fn GSelf(self);
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
  let unused x: () = T.F();
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
  // CHECK:STDERR: fail_name_lookup_through_where_self_impls.carbon:[[@LINE+4]]:3: error: member name `G` not found in `I` [MemberNameNotFoundInSpecificScope]
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
  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:6: error: member name `Bfn` not found in `A` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   T.((A where .Self impls B).Bfn)();
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.((A where .Self impls B).Bfn)();

  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:3: error: member name `Bfn` not found in `A` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   (A where .Self impls B).Bfn;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (A where .Self impls B).Bfn;
}

// --- fail_name_lookup_instance.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf(self);
}

interface J {
  fn GNonInstance();
  fn GSelf(self);
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GNonInstance` not found in `I` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   x.GNonInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  x.GNonInstance();

  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GSelf` not found in `I` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   x.GSelf();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.GSelf();
}

// --- compare_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
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
}

interface I {}
interface J {}
interface K {}

// --- fail_todo_compare_equal_with_rewrite.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I { let A:! type; }
interface J { let B:! type; }

// TODO: The `.Self` implied in `.A` contains all of the extended interfaces,
// which is different between these facet types at the point where `.A` is
// written. So the resulting facet types are different. We should drop
// interfaces from the rewrite constraint's LHS that don't apply so that we get
// a canonical representation of `.A` that is independent of the current state
// of `.Self` since `.A = ()` can not be written as, and is thus not equivalent
// to, `.Self.(I.A) = ()`.

fn Test() {
  // TODO: This should pass.
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   AssertSame(Type((I where .A = ()) & J),
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE-19]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn AssertSame[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AssertSame(Type((I where .A = ()) & J),
             Type(I & J where .A = ()));
  // TODO: This should pass.
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   AssertSame(Type(I & (J where .B = {})),
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE-29]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn AssertSame[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AssertSame(Type(I & (J where .B = {})),
             Type(I & J where .B = {}));

  // TODO: This should pass.
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   AssertSame(Type((I where .A = ()) & (J where .B = {})),
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_compare_equal_with_rewrite.carbon:[[@LINE-40]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn AssertSame[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  AssertSame(Type((I where .A = ()) & (J where .B = {})),
             Type(I & J where .A = () and .B = {}));
}

// --- fail_compare_not_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn Same[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I {}
interface J {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   Same(Type(I where .Self impls J), Type(J where .Self impls I));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Same[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Same(Type(I where .Self impls J), Type(J where .Self impls I));
}

// --- todo_fail_compare_not_equal_with_same_type.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn Same[T:! type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I { let A:! type; }
interface J { let B:! type; }

fn Test() {
  // The first `.A` has a `.Self` of type `I`. The second contains a `.Self` of
  // type `I & J`. That makes the two constaints `.A == ()` have different
  // constant values, so these are different facet types.
  //
  // TODO: This should fail. We don't store same-type constraints in the facet type yet.
  Same(Type((I where .A == ()) & J), Type(I & J where .A == ()));
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

// --- impl_with_rewrite_of_interface_not_being_implemented.carbon
library "[[@TEST_NAME]]";

interface I { let A:! type; }
interface J { let A:! type; }

class C {}

// Implementation of `C as J`.
impl C as J where .A = {} {}

// Requirement of implementing J with `.A = A`.
constraint NeedJ(A:! type) {
  require impls J where .A = A;
}

// This is an implementation of `I` with `I.A = ()`. The requirement
// that `C` also impls `J where .A = {}` is an additional constraint
// that must be satisfied (and is satisfied by the impl above, though
// that currently isn't checked), but doesn't affect `C as I`.
impl C as I where .A = () and .Self impls NeedJ({}) {}

let x: C.(I.A) = ();
let y: C.(J.A) = {};
