// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/effectively_final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/effectively_final.carbon

// --- effectively_final.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

// An impl is effectively final if all of its generic bindings are of the
// following types:

// - `bool`

impl forall [B:! bool] () as A(B) where .Result = {} {}
fn Bool(B:! bool) -> (() as A(B)).Result { return {}; }

// - character literal type

impl forall [N:! Core.CharLiteral()] () as A(N) where .Result = {} {}
fn CharLiteral(N:! Core.CharLiteral()) -> (() as A(N)).Result { return {}; }

// - integer type

impl forall [N:! i32] () as A(N) where .Result = {} {}
fn Int(N:! i32) -> (() as A(N)).Result { return {}; }

// - integer literal type

impl forall [N:! Core.IntLiteral()] () as A(N) where .Result = {} {}
fn IntLiteral(N:! Core.IntLiteral()) -> (() as A(N)).Result { return {}; }

// - floating-point type

impl forall [N:! f64] () as A(N) where .Result = {} {}
fn Float(N:! f64) -> (() as A(N)).Result { return {}; }

// - floating-point literal type

impl forall [N:! Core.FloatLiteral()] () as A(N) where .Result = {} {}
fn FloatLiteral(N:! Core.FloatLiteral()) -> (() as A(N)).Result { return {}; }

// - array of one of the above

impl forall [Arr:! array(i32, 4)] () as A(Arr) where .Result = {} {}
fn Array(Arr:! array(i32, 4)) -> (() as A(Arr)).Result { return {}; }

// - struct containing the above

impl forall [S:! {.a: i32, .b: f64}] () as A(S) where .Result = {} {}
fn Struct(S:! {.a: i32, .b: f64}) -> (() as A(S)).Result { return {}; }

// - tuple containing the above

impl forall [T:! (i32, f64)] () as A(T) where .Result = {} {}
fn Tuple(T:! (i32, f64)) -> (() as A(T)).Result { return {}; }

// - (recursively)

impl forall [T:! (i32, {.x: array(f64, 4)})] () as A(T) where .Result = {} {}
fn Nested(T:! (i32, {.x: array(f64, 4)})) -> (() as A(T)).Result { return {}; }

// --- fail_not_effectively_final_type.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

// An impl with a binding of any other type is not effectively final, as it
// could be further specialized.

impl forall [T:! type] () as A(T) where .Result = {} {}
// CHECK:STDERR: fail_not_effectively_final_type.carbon:[[@LINE+7]]:44: error: cannot implicitly convert expression of type `{}` to `().(A(T).Result)` [ConversionFailure]
// CHECK:STDERR: fn Type(T:! type) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                            ^~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_type.carbon:[[@LINE+4]]:44: note: type `{}` does not implement interface `Core.ImplicitAs(().(A(T).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn Type(T:! type) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                            ^~~~~~~~~~
// CHECK:STDERR:
fn Type(T:! type) -> (() as A(T)).Result { return {}; }

// --- fail_not_effectively_final_facet.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

interface B {}

impl forall [T:! B] () as A(T) where .Result = {} {}
// CHECK:STDERR: fail_not_effectively_final_facet.carbon:[[@LINE+7]]:41: error: cannot implicitly convert expression of type `{}` to `().(A(T).Result)` [ConversionFailure]
// CHECK:STDERR: fn Type(T:! B) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                         ^~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_facet.carbon:[[@LINE+4]]:41: note: type `{}` does not implement interface `Core.ImplicitAs(().(A(T).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn Type(T:! B) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                         ^~~~~~~~~~
// CHECK:STDERR:
fn Type(T:! B) -> (() as A(T)).Result { return {}; }

// --- fail_not_effectively_final_nested.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

impl forall [T:! (i32, {.x: type})] () as A(T) where .Result = {} {}
// CHECK:STDERR: fail_not_effectively_final_nested.carbon:[[@LINE+7]]:57: error: cannot implicitly convert expression of type `{}` to `().(A(T).Result)` [ConversionFailure]
// CHECK:STDERR: fn Type(T:! (i32, {.x: type})) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                                         ^~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_nested.carbon:[[@LINE+4]]:57: note: type `{}` does not implement interface `Core.ImplicitAs(().(A(T).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn Type(T:! (i32, {.x: type})) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                                         ^~~~~~~~~~
// CHECK:STDERR:
fn Type(T:! (i32, {.x: type})) -> (() as A(T)).Result { return {}; }

// --- fail_not_effectively_final_incomplete.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

// We can't tell whether a parameter of an incomplete type would make the impl
// effectively final, but it doesn't matter because the impl is invalid anyway.
class Incomplete;

// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE+10]]:36: error: forming value of incomplete type `Incomplete` [IncompleteTypeInValueConversion]
// CHECK:STDERR: impl forall [T:! Incomplete] () as A(T) where .Result = {} {}
// CHECK:STDERR:                                    ^~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: interface A[T:! type](V:! T) { let Result:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Incomplete] () as A(T) where .Result = {} {}
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE+10]]:35: error: forming value of incomplete type `Incomplete` [IncompleteTypeInValueConversion]
// CHECK:STDERR: fn Type(T:! Incomplete) -> (() as A(T)).Result { return {}; }
// CHECK:STDERR:                                   ^~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-16]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_effectively_final_incomplete.carbon:[[@LINE-23]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: interface A[T:! type](V:! T) { let Result:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Type(T:! Incomplete) -> (() as A(T)).Result { return {}; }

// --- fail_specialized_after_concrete_use.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

impl forall [B:! bool] () as A(B) where .Result = {} {}

// CHECK:STDERR: fail_specialized_after_concrete_use.carbon:[[@LINE+3]]:15: error: found `impl` that would change the result of an earlier use of `() as A(true)` [PoisonedImplLookupConcreteResult]
// CHECK:STDERR: fn Bool() -> (() as A(true)).Result { return {}; }
// CHECK:STDERR:               ^~~~~~~~~~~~~
fn Bool() -> (() as A(true)).Result { return {}; }

// CHECK:STDERR: fail_specialized_after_concrete_use.carbon:[[@LINE+7]]:1: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: impl () as A(true) where .Result = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_specialized_after_concrete_use.carbon:[[@LINE-10]]:1: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR: impl forall [B:! bool] () as A(B) where .Result = {} {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A(true) where .Result = () {}

// --- fail_specialized_after_generic_use.carbon

library "[[@TEST_NAME]]";

interface A[T:! type](V:! T) { let Result:! type; }

impl forall [B:! bool] () as A(B) where .Result = {} {}

fn Bool(B:! bool) -> (() as A(B)).Result { return {}; }

impl () as A(true) where .Result = () {}
