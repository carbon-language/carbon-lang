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
}

// --- same.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {
  fn G();
}

class C {}
impl C as A {
  fn G() {}
}

fn F() {
  ({} as C).((A & A).G)();
  (({} as C) as (C as (A & A))).((A & A).G)();
  (({} as C) as (C as (A & A))).(A.G)();
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

// --- fail_todo_combine.carbon
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
  ({} as C).((A & B).G)();

  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+11]]:18: error: semantics TODO: `impl lookup for a FacetType with more than one interface` [SemanticsTodo]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+7]]:18: error: cannot convert value of type `type` to `A & B` with `as` [ExplicitAsConversionFailure]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+4]]:18: note: type `type` does not implement interface `Core.As(A & B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((A & B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & B))).((A & B).G)();

  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+11]]:18: error: semantics TODO: `impl lookup for a FacetType with more than one interface` [SemanticsTodo]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+7]]:18: error: cannot convert value of type `type` to `A & B` with `as` [ExplicitAsConversionFailure]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_combine.carbon:[[@LINE+4]]:18: note: type `type` does not implement interface `Core.As(A & B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (({} as C) as (C as (A & B))).((B).G)();
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & B))).((B).G)();
}

// --- fail_todo_generic_combine.carbon

library "[[@TEST_NAME]]";

import Core;

interface A {
}
interface B {
  fn G();
}

fn G[T:! A & B](t: T) {
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE+4]]:3: error: cannot access member of interface `B` in type `T` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   t.(B.G)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  t.(B.G)();
}

class C {}
impl C as A {}
impl C as B {
  fn G() {}
}

fn F() {
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE+17]]:3: error: semantics TODO: `impl lookup for a FacetType with more than one interface` [SemanticsTodo]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE-18]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A & B](t: T) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE+10]]:3: error: cannot implicitly convert value of type `type` to `A & B` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(A & B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_combine.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A & B](t: T) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);
}
