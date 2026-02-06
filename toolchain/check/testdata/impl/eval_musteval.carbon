// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/eval_musteval.carbon

// --- interface.carbon

library "[[@TEST_NAME]]";

interface Runtime {
  fn F[self: Self]() -> type;
}

interface Eval {
  eval fn F[self: Self]() -> type;
}

interface MustEval {
  musteval fn F[self: Self]() -> type;
}

// --- impl_runtime.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A { adapt {}; }

impl A as Runtime {
  fn F[self: Self]() -> type { return A; }
}

impl A as Eval {
  // TODO: Consider rejecting this, as compile-time evaluation would always
  // fail.
  fn F[self: Self]() -> type { return A; }
}

impl A as MustEval {
  // TODO: Consider rejecting this, as compile-time evaluation would always
  // fail.
  fn F[self: Self]() -> type { return A; }
}

fn Call() {
  ({} as A).(Runtime.F)();
  ({} as A).(Eval.F)();
  ({} as A).(MustEval.F)();
}

// --- fail_impl_runtime_eval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A { adapt {}; }

impl A as Runtime {
  fn F[self: Self]() -> type { return A; }
}

impl A as Eval {
  fn F[self: Self]() -> type { return A; }
}

impl A as MustEval {
  fn F[self: Self]() -> type { return A; }
}

fn Call() {
  // CHECK:STDERR: fail_impl_runtime_eval.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   let t: ({} as A).(Runtime.F)() = {} as A;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let t: ({} as A).(Runtime.F)() = {} as A;
  // CHECK:STDERR: fail_impl_runtime_eval.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   let u: ({} as A).(Eval.F)() = {} as A;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let u: ({} as A).(Eval.F)() = {} as A;
  // CHECK:STDERR: fail_impl_runtime_eval.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   let v: ({} as A).(MustEval.F)() = {} as A;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let v: ({} as A).(MustEval.F)() = {} as A;
}

// --- impl_eval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A { adapt {}; }

impl A as Runtime {
  eval fn F[self: Self]() -> type { return A; }
}

impl A as Eval {
  eval fn F[self: Self]() -> type { return A; }
}

impl A as MustEval {
  eval fn F[self: Self]() -> type { return A; }
}

fn Call() {
  // TODO: Should we allow calling this at compile time? This is not allowed if
  // we generate a thunk; see the next split for an example.
  let t: ({} as A).(Runtime.F)() = {} as A;
  let u: ({} as A).(Eval.F)() = {} as A;
  let v: ({} as A).(MustEval.F)() = {} as A;
}

// --- fail_todo_impl_eval_from_runtime_thunk.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B {}

class BView {}
impl B as Core.ImplicitAs(BView) {
  fn Convert[self: Self]() -> BView { return {}; }
}

impl B as Runtime {
  eval fn F[self: BView]() -> type { return B; }
}

fn Call() {
  // TODO: Should we allow calling this at compile time? This is allowed if we
  // don't generate a thunk.
  // CHECK:STDERR: fail_todo_impl_eval_from_runtime_thunk.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   let t: ({} as B).(Runtime.F)() = {} as B;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let t: ({} as B).(Runtime.F)() = {} as B;
}

// --- fail_impl_musteval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B { adapt {}; }

impl B as Runtime {
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+11]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   musteval fn F[self: Self]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: Self]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:5:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self]() -> type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: Self]() -> type { return B; }
}

impl B as Eval {
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+11]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   musteval fn F[self: Self]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: Self]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE-26]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:9:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   eval fn F[self: Self]() -> type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: Self]() -> type { return B; }
}

// --- impl_musteval_from_musteval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B { adapt {}; }

impl B as MustEval {
  musteval fn F[self: Self]() -> type { return B; }
}

fn Call() {
  let t: ({} as B).(MustEval.F)() = {} as B;
}

// --- fail_todo_impl_musteval_from_musteval_thunk.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B {}

class BView {}
impl B as Core.ImplicitAs(BView) {
  fn Convert[self: Self]() -> BView { return {}; }
}

impl B as MustEval {
  // CHECK:STDERR: fail_todo_impl_musteval_from_musteval_thunk.carbon:[[@LINE+11]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   musteval fn F[self: BView]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_musteval_from_musteval_thunk.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: BView]() -> type { return B; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_musteval_from_musteval_thunk.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:13:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   musteval fn F[self: Self]() -> type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: BView]() -> type { return B; }
}
