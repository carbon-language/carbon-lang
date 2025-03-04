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
  // CHECK:STDERR: core.carbon:[[@LINE+8]]:36: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   fn Convert[self: Self]() -> Dest;!
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR:
  // CHECK:STDERR: core.carbon:[[@LINE+4]]:36: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   fn Convert[self: Self]() -> Dest;!
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR:
  fn Convert[self: Self]() -> Dest;!
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

// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn AC[T:! A & C](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn AC[T:! A & C](t: T) {}
// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn CA[T:! C & A](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn CA[T:! C & A](t: T) {}
// CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:11: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn CC[T:! C & C](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn CC[T:! C & C](t: T) {}

fn F() {
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   ({} as C) as (C as (A & C));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (A & C));
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   ({} as C) as (C as (C & A));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (C & A));
  // CHECK:STDERR: fail_combine_with_non_facet_type.carbon:[[@LINE+4]]:23: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
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

// CHECK:STDERR: same.carbon:[[@LINE+4]]:11: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn AA[T:! A & A](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn AA[T:! A & A](t: T) {}

fn F() {
  // CHECK:STDERR: same.carbon:[[@LINE+4]]:23: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   ({} as C) as (C as (A & A));
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
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

  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE+4]]:15: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   ({} as C).((A & B).G)();
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR:
  ({} as C).((A & B).G)();
}

// --- combine.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {}
interface B {
  fn G();
}

class C {}
impl C as A {}
impl C as B {
  fn G() {}
}

fn F() {
  // CHECK:STDERR: combine.carbon:[[@LINE+4]]:15: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   ({} as C).((A & B).G)();
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR:
  ({} as C).((A & B).G)();

  // CHECK:STDERR: combine.carbon:[[@LINE+8]]:24: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).G)();
  // CHECK:STDERR:                        ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: combine.carbon:[[@LINE+4]]:35: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).G)();
  // CHECK:STDERR:                                   ^~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & B))).((A & B).G)();

  // CHECK:STDERR: combine.carbon:[[@LINE+4]]:24: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((B).G)();
  // CHECK:STDERR:                        ^~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & B))).((B).G)();
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

// CHECK:STDERR: generic_interface.carbon:[[@LINE+4]]:10: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn G[T:! A(P1) & B](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR:
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

// CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+4]]:10: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn G[T:! A(P2) & B](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR:
fn G[T:! A(P2) & B](t: T) {}

fn F() {
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

// CHECK:STDERR: generic_forall_impl.carbon:[[@LINE+4]]:10: error: name `Core.BitAnd` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn G[T:! Iface & GenericIface(GenericClass(ImplIface))](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G[T:! Iface & GenericIface(GenericClass(ImplIface))](t: T) {}

fn F() {
  G({} as C);
}
