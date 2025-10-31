// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon

// --- fail_incomplete_interface.carbon
library "[[@TEST_NAME]]";

interface A;
interface B {}
class C {}

fn G[T:! A](t: T) {}
fn H[T:! A & B](t: T) {}

fn F() {
  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE+4]]:3: error: cannot convert type `C` into type implementing `A` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C as A;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  C as A;

  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE+4]]:3: error: cannot convert type `C` into type implementing `A & B` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C as (A & B);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  C as (A & B);

  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `A` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE-19]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);

  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `A & B` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   H({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE-27]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn H[T:! A & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  H({} as C);
}

// --- fail_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint A;
interface B {}
class C {}

fn G[T:! A](t: T) {}
fn H[T:! A & B](t: T) {}

fn F() {
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+7]]:3: error: facet type `A` is incomplete [ImplLookupInIncompleteFacetType]
  // CHECK:STDERR:   C as A;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-11]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint A;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C as A;

  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+7]]:3: error: facet type `B & A` is incomplete [ImplLookupInIncompleteFacetType]
  // CHECK:STDERR:   C as (A & B);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-20]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint A;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C as (A & B);

  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+10]]:3: error: facet type `A` is incomplete [ImplLookupInIncompleteFacetType]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-29]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint A;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);

  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+10]]:3: error: facet type `B & A` is incomplete [ImplLookupInIncompleteFacetType]
  // CHECK:STDERR:   H({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-41]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint A;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn H[T:! A & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  H({} as C);
}
