// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/eval_musteval.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn Basic() -> i32 { return 0; }
eval fn Eval() -> i32 { return 0; }
musteval fn MustEval() -> i32 { return 0; }

// --- member.carbon

library "[[@TEST_NAME]]";

class C {
  eval fn Eval() {}
  musteval fn MustEval() {}
}

// --- private.carbon

library "[[@TEST_NAME]]";

private eval fn Eval() -> i32 { return 0; }
private musteval fn MustEval() -> i32 { return 0; }

// --- fail_conflict.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+3]]:6-13: error: `musteval` not allowed on declaration with `eval` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+2]]:1-4: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
eval musteval fn Mixed();

// --- fail_order.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_order.carbon:[[@LINE+3]]:6-11: error: `export` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: fail_order.carbon:[[@LINE+2]]:1-4: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
eval export fn ExportEval();

// CHECK:STDERR: fail_order.carbon:[[@LINE+3]]:6-12: error: `private` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: fail_order.carbon:[[@LINE+2]]:1-4: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
eval private fn PrivateEval();

class C {
  // CHECK:STDERR: fail_order.carbon:[[@LINE+3]]:8-14: error: `virtual` must appear before `eval` [ModifierMustAppearBefore]
  // CHECK:STDERR: fail_order.carbon:[[@LINE+2]]:3-6: note: `eval` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:
  eval virtual fn VirtualEval();
}

// --- fail_not_fn.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+2]]:1-4: error: `eval` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
eval var x: i32;

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+2]]:1-4: error: `eval` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
eval class C {};

// --- redecl.carbon

library "[[@TEST_NAME]]";

eval fn A() -> i32;
eval fn A() -> i32 { return 0; }

musteval fn B() -> i32;
musteval fn B() -> i32 { return 0; }

// --- fail_differ_in_redecl.carbon

library "[[@TEST_NAME]]";

fn NoneThenEval();
fn NoneThenMustEval();
eval fn EvalThenNone();
eval fn EvalThenMustEval();
musteval fn MustEvalThenNone();
musteval fn MustEvalThenEval();

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-24: error: function redeclaration differs because new function is `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-8]]:1-18: note: previously not declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
eval fn NoneThenEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-32: error: function redeclaration differs because new function is `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-12]]:1-22: note: previously not declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
musteval fn NoneThenMustEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-19: error: function redeclaration differs because new function is not `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-16]]:1-23: note: previously declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
fn EvalThenNone() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-32: error: function redeclaration differs because new function is `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-20]]:1-27: note: previously declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
musteval fn EvalThenMustEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-23: error: function redeclaration differs because new function is not `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-24]]:1-31: note: previously declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
fn MustEvalThenNone() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+3]]:1-28: error: function redeclaration differs because new function is `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-28]]:1-31: note: previously declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:
eval fn MustEvalThenEval() {}
