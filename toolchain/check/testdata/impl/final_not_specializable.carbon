// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/final_not_specializable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/final_not_specializable.carbon

// --- interface.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {}

// --- not_specializable.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

// An impl is allowed to be declared as final if all of its generic bindings are
// of the following types:

// - `bool`

final impl forall [B:! bool] () as I(A(B)) {}

// - character literal type

final impl forall [N:! Core.CharLiteral()] () as I(A(N)) {}

// - integer type

final impl forall [N:! i32] () as I(A(N)) {}

// - integer literal type

final impl forall [N:! Core.IntLiteral()] () as I(A(N)) {}

// - floating-point type

final impl forall [N:! f64] () as I(A(N)) {}

// - floating-point literal type

final impl forall [N:! Core.FloatLiteral()] () as I(A(N)) {}

// - array of one of the above

final impl forall [Arr:! array(i32, 4)] () as I(A(Arr)) {}

// - struct containing the above

final impl forall [S:! {.a: i32, .b: f64}] () as I(A(S)) {}

// - tuple containing the above

final impl forall [T:! (i32, f64)] () as I(A(T)) {}

// - (recursively)

final impl forall [T:! (i32, {.x: array(f64, 4)})] () as I(A(T)) {}

// --- fail_specializable_type.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

// An impl with a binding of any other type can't be declared `final` except in
// the library that owns its root type or interface, as it could be further
// specialized.

// CHECK:STDERR: fail_specializable_type.carbon:[[@LINE+4]]:1: error: `final impl` found in file that does not contain the root self type nor the interface definition, and might be specialized in other files [FinalImplInvalidFile]
// CHECK:STDERR: final impl forall [T:! type] () as I(A(T)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] () as I(A(T)) {}

// --- fail_specializable_facet.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

interface B {}

// CHECK:STDERR: fail_specializable_facet.carbon:[[@LINE+4]]:1: error: `final impl` found in file that does not contain the root self type nor the interface definition, and might be specialized in other files [FinalImplInvalidFile]
// CHECK:STDERR: final impl forall [T:! B] () as I(A(T)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! B] () as I(A(T)) {}

// --- fail_specializable_nested.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

// CHECK:STDERR: fail_specializable_nested.carbon:[[@LINE+4]]:1: error: `final impl` found in file that does not contain the root self type nor the interface definition, and might be specialized in other files [FinalImplInvalidFile]
// CHECK:STDERR: final impl forall [T:! (i32, {.x: type})] () as I(A(T)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! (i32, {.x: type})] () as I(A(T)) {}

// --- fail_not_effectively_final_incomplete.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

// We can't tell whether a parameter of an incomplete type would make the impl
// effectively final, but it doesn't matter because the impl is invalid anyway.
class Incomplete;

// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE+10]]:44: error: forming value of incomplete type `Incomplete` [IncompleteTypeInValueConversion]
// CHECK:STDERR: final impl forall [T:! Incomplete] () as I(A(T)) {}
// CHECK:STDERR:                                            ^~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class A[T:! type](V:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! Incomplete] () as I(A(T)) {}

// --- fail_final_specialized.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A[T:! type](V:! T) {}

final impl forall [B:! bool] () as I(A(B)) {}

// CHECK:STDERR: fail_final_specialized.carbon:[[@LINE+7]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl () as I(A(true)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_specialized.carbon:[[@LINE-5]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [B:! bool] () as I(A(B)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I(A(true)) {}
