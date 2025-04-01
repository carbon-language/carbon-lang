// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/combine.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/combine.carbon

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

  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE+4]]:14: error: ambiguous use of name `G` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   ({} as C).((A & B).G)();
  // CHECK:STDERR:              ^~~~~~~~~
  // CHECK:STDERR:
  ({} as C).((A & B).G)();
}

// --- combine.carbon
library "[[@TEST_NAME]]";

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

interface A(T:! type) {}
interface B {}

class P1 {}
class P2 {}

class C {}
impl C as A(P1) {}
impl C as B {}

fn G[T:! A(P2) & B](t: T) {}

fn F() {
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `A(P2) & B` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A(P2) & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);
}

// --- generic_forall_impl.carbon
library "[[@TEST_NAME]]";

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
