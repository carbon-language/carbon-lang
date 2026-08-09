// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/combine.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/combine.carbon

// --- fail_name_collision.carbon
library "[[@TEST_NAME]]";

interface A {
  fn G();
}
interface B {
  fn G();
}

class C {}
impl C as A {
  fn G();
}
impl C as B {
  fn G() {}
}

fn F() {
  // TODO: This error message is wrong here, we are not using `extend`.

  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE+4]]:14-22: error: ambiguous use of name `G` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE-15]]:3-9: note: candidate `G` declared here [NameAmbiguousCandidate]
  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE-19]]:3-9: note: candidate `G` declared here [NameAmbiguousCandidate]
  // CHECK:STDERR:
  ({} as C).((A & B).G)();
}

// --- combine.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {
  fn BB(self);
}

class C {}
impl C as A {}
impl C as B {
  fn BB(unused self) {}
}

fn G[T: A & B](unused t: T) {}

fn F() {
  ({} as C).((A & B).BB)();

  (({} as C) as (C as (A & B))).((A & B).BB)();
  (({} as C) as (C as (A & B))).(B.BB)();

  G({} as C);
}

// --- generic_interface.carbon
library "[[@TEST_NAME]]";

interface A(T: type) {}
interface B {}

class P1 {}
class P2 {}

class C {}
impl C as A(P1) {}
impl C as B {}

fn G[T: A(P1) & B](unused t: T) {}

fn F() {
  G({} as C);
}

// --- fail_wrong_generic_interface.carbon
library "[[@TEST_NAME]]";

interface A(T: type) {}
interface B {}

class P1 {}
class P2 {}

class C {}
impl C as A(P1) {}
impl C as B {}

fn G[T: A(P2) & B](unused t: T) {}

fn F() {
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+3]]:3-12: error: cannot convert type `C` into type implementing `A(P2) & B` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE-4]]:1-33: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  G({} as C);
}

// --- generic_forall_impl.carbon
library "[[@TEST_NAME]]";

interface Iface {}
interface GenericIface(T: type) {}

class GenericClass(T: type) {}

class ImplIface {}
impl ImplIface as Iface {}

class C {}
impl C as Iface {}

impl forall [IfaceType: Iface] C as GenericIface(GenericClass(IfaceType)) {}

fn G[T: Iface & GenericIface(GenericClass(ImplIface))](unused t: T) {}

fn F() {
  G({} as C);
}

// --- compare_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T: type) {}
fn AssertSame[T: type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(generic T: type) -> WrapType(T) { return {}; }

interface I;
interface J;
interface K(T: type);

fn TestIncomplete() {
  AssertSame(Type(I), Type(I & I));
  AssertSame(Type(I), Type(I & I & I));
  AssertSame(Type(I & J), Type(J & I));
  AssertSame(Type(I & J), Type(I & I & J));
  AssertSame(Type(I & J), Type(I & J & I));
  AssertSame(Type(I & J), Type(J & I & I));
  AssertSame(Type(I & K({})), Type(K({}) & I));
  AssertSame(Type(I & K({}) & K(())), Type(K(()) & K({}) & I));
}

interface I {}
interface J {}
interface K(T: type) {}

fn TestComplete() {
  AssertSame(Type(I), Type(I & I));
  AssertSame(Type(I), Type(I & I & I));
  AssertSame(Type(I & J), Type(J & I));
  AssertSame(Type(I & J), Type(I & I & J));
  AssertSame(Type(I & J), Type(I & J & I));
  AssertSame(Type(I & J), Type(J & I & I));
  AssertSame(Type(I & K({})), Type(K({}) & I));
  AssertSame(Type(I & K({}) & K(())), Type(K(()) & K({}) & I));
}

// --- fail_compare_not_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T: type) {}
fn Same[T: type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(generic T: type) -> WrapType(T) { return {}; }

interface I {}
interface J {}
interface K {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+4]]:3-36: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+3]]:21-35: note: conflicting value for `T` deduced from this argument [DeductionInconsistentArg]
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE-10]]:1-64: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Same(Type(I & J), Type(K & I & J));
}

// --- fail_compare_not_equal_parameterized.carbon
library "[[@TEST_NAME]]";

class WrapType(T: type) {}
fn Same[T: type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(generic T: type) -> WrapType(T) { return {}; }

interface I {}
interface J(T: type) {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal_parameterized.carbon:[[@LINE+4]]:3-40: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR: fail_compare_not_equal_parameterized.carbon:[[@LINE+3]]:25-39: note: conflicting value for `T` deduced from this argument [DeductionInconsistentArg]
  // CHECK:STDERR: fail_compare_not_equal_parameterized.carbon:[[@LINE-9]]:1-64: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Same(Type(I & J(())), Type(J({}) & I));
}

// --- fail_compare_not_equal_parameterized_extra.carbon
library "[[@TEST_NAME]]";

class WrapType(T: type) {}
fn Same[T: type](unused a: WrapType(T), unused b: WrapType(T)) {}
fn Type(generic T: type) -> WrapType(T) { return {}; }

interface I {}
interface J(T: type) {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal_parameterized_extra.carbon:[[@LINE+4]]:3-48: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR: fail_compare_not_equal_parameterized_extra.carbon:[[@LINE+3]]:25-47: note: conflicting value for `T` deduced from this argument [DeductionInconsistentArg]
  // CHECK:STDERR: fail_compare_not_equal_parameterized_extra.carbon:[[@LINE-9]]:1-64: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Same(Type(I & J(())), Type(J(()) & J({}) & I));
}
