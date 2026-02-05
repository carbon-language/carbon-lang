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
  fn F[self: Self]();
}

interface Eval {
  eval fn F[self: Self]();
}

interface MustEval {
  musteval fn F[self: Self]();
}

// --- impl_runtime.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A {}

impl A as Runtime {
  fn F[self: Self]() {}
}

impl A as Eval {
  fn F[self: Self]() {}
}

impl A as MustEval {
  fn F[self: Self]() {}
}

// --- impl_eval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class A {}

impl A as Runtime {
  eval fn F[self: Self]() {}
}

impl A as Eval {
  eval fn F[self: Self]() {}
}

impl A as MustEval {
  eval fn F[self: Self]() {}
}

// --- fail_impl_musteval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B {}

impl B as Runtime {
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+11]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   musteval fn F[self: Self]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: Self]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:5:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: Self]() {}
}

impl B as Eval {
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+11]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   musteval fn F[self: Self]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: Self]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_musteval.carbon:[[@LINE-26]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:9:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   eval fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: Self]() {}
}

// --- impl_musteval_from_musteval.carbon

library "[[@TEST_NAME]]";

import library "interface";

class B {}

impl B as MustEval {
  musteval fn F[self: Self]() {}
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
  // CHECK:STDERR:   musteval fn F[self: BView]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_musteval_from_musteval_thunk.carbon:[[@LINE+8]]:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   musteval fn F[self: BView]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_musteval_from_musteval_thunk.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: interface.carbon:13:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   musteval fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  musteval fn F[self: BView]() {}
}
