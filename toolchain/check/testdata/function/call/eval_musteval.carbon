// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/eval_musteval.carbon

// --- no_eval.carbon

library "[[@TEST_NAME]]";

fn Basic(n: i32) {}
eval fn Eval(n: i32) {}

fn RuntimeArg(n: i32) {
  Basic(n);
  Eval(n);
}

// --- fail_no_eval_musteval.carbon

library "[[@TEST_NAME]]";

musteval fn MustEval(n: i32) {}

fn RuntimeArg(n: i32) {
  // CHECK:STDERR: fail_no_eval_musteval.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   MustEval(n);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_no_eval_musteval.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: musteval fn MustEval(n: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  MustEval(n);
}

// --- eval.carbon

library "[[@TEST_NAME]]";

fn Basic(n: i32) {}
eval fn Eval(n: i32) {}
musteval fn MustEval(n: i32) {}

fn CompileTimeArg() {
  Basic(1);
  Eval(1);
  MustEval(1);
}

// --- fail_todo_musteval_calls_musteval.carbon

library "[[@TEST_NAME]]";

musteval fn MustEval(n: i32) {}

// TODO: This should be accepted.
musteval fn CallMustEval(n: i32) {
  // CHECK:STDERR: fail_todo_musteval_calls_musteval.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   MustEval(n);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_musteval_calls_musteval.carbon:[[@LINE-7]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: musteval fn MustEval(n: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  MustEval(n);
}
