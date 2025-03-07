// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/no_prelude/combine.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/no_prelude/combine.carbon

// --- core.carbon
package Core;

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

impl forall [T:! type] T as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}

// --- fail_combine_with_non_facet_type.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {}
class C {}

// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn AC[T:! A & C](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn AC[T:! A & C](t: T) {}
// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn CA[T:! C & A](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn CA[T:! C & A](t: T) {}
// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+8]]:11: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn CC[T:! C & C](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn CC[T:! C & C](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn CC[T:! C & C](t: T) {}

fn F() {
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
  // CHECK:STDERR:   ({} as C) as (C as (A & C));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (A & C));
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
  // CHECK:STDERR:   ({} as C) as (C as (C & A));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (C & A));
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+8]]:23: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
  // CHECK:STDERR:   ({} as C) as (C as (C & C));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: non-facet type `C` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
  // CHECK:STDERR:   ({} as C) as (C as (C & C));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (C & C));

  AC({} as C);
  CA({} as C);
  CC({} as C);
}

// --- same.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {}

class C {}
impl C as A {}

fn AA[T:! A & A](t: T) {}

fn F() {
  ({} as C) as (C as (A & A));

  AA({} as C);
}

// --- fail_name_collision.carbon
library "[[@TEST_NAME]]";

import Core;

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

  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE+4]]:14: error: ambiguous use of name `G` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   ({} as C).((A & B).G)();
  // CHECK:STDERR:              ^~~~~~~~~
  // CHECK:STDERR:
  ({} as C).((A & B).G)();
}

// --- combine.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {}
interface B {
  fn BB[self: Self]();
}

class C {}
impl C as A {}
impl C as B {
  fn BB[self: Self]() {}
}

fn G[T:! A & B](t: T) {}

fn F() {
  ({} as C).((A & B).BB)();

  (({} as C) as (C as (A & B))).((A & B).BB)();
  (({} as C) as (C as (A & B))).(B.BB)();

  G({} as C);
}

// --- generic_interface.carbon
library "[[@TEST_NAME]]";

import Core;

interface A(T:! type) {}
interface B {}

class P1 {}
class P2 {}

class C {}
impl C as A(P1) {}
impl C as B {}

fn G[T:! A(P1) & B](t: T) {}

fn F() {
  G({} as C);
}

// --- fail_wrong_generic_interface.carbon
library "[[@TEST_NAME]]";

import Core;

interface A(T:! type) {}
interface B {}

class P1 {}
class P2 {}

class C {}
impl C as A(P1) {}
impl C as B {}

fn G[T:! A(P2) & B](t: T) {}

fn F() {
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+10]]:3: error: cannot implicitly convert value of type `type` to `A(P2) & B` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(A(P2) & B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A(P2) & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);
}

// --- generic_forall_impl.carbon
library "[[@TEST_NAME]]";

import Core;

interface Iface {}
interface GenericIface(T:! type) {}

class GenericClass(T:! type) {}

class ImplIface {}
impl ImplIface as Iface {}

class C {}
impl C as Iface {}

impl forall [IfaceType:! Iface] C as GenericIface(GenericClass(IfaceType)) {}

fn G[T:! Iface & GenericIface(GenericClass(ImplIface))](t: T) {}

fn F() {
  G({} as C);
}
