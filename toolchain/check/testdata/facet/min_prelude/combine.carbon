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

  // CHECK:STDERR: fail_name_collision.carbon:[[@LINE+7]]:15: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   ({} as C).((A & B).G)();
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface BitAnd {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
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

// CHECK:STDERR: combine.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G[T:! A & B](t: T) {}
// CHECK:STDERR:          ^~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G[T:! A & B](t: T) {}

fn F() {
  // CHECK:STDERR: combine.carbon:[[@LINE+7]]:15: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   ({} as C).((A & B).BB)();
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface BitAnd {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C).((A & B).BB)();

  // CHECK:STDERR: combine.carbon:[[@LINE+14]]:24: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).BB)();
  // CHECK:STDERR:                        ^~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface BitAnd {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: combine.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).BB)();
  // CHECK:STDERR:                                   ^~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface BitAnd {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & B))).((A & B).BB)();
  // CHECK:STDERR: combine.carbon:[[@LINE+7]]:24: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).(B.BB)();
  // CHECK:STDERR:                        ^~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface BitAnd {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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

// CHECK:STDERR: generic_interface.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G[T:! A(P1) & B](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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

// CHECK:STDERR: fail_wrong_generic_interface.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G[T:! A(P2) & B](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G[T:! A(P2) & B](t: T) {}

fn F() {
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

// CHECK:STDERR: generic_forall_impl.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G[T:! Iface & GenericIface(GenericClass(ImplIface))](t: T) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G[T:! Iface & GenericIface(GenericClass(ImplIface))](t: T) {}

fn F() {
  G({} as C);
}
