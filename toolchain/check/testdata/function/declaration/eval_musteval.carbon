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

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+7]]:6: error: `musteval` not allowed on declaration with `eval` [ModifierNotAllowedWith]
// CHECK:STDERR: eval musteval fn Mixed();
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval musteval fn Mixed();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval musteval fn Mixed();

// --- fail_order.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:6: error: `export` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: eval export fn ExportEval();
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval export fn ExportEval();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval export fn ExportEval();

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:6: error: `private` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: eval private fn PrivateEval();
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval private fn PrivateEval();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval private fn PrivateEval();

class C {
  // CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:8: error: `virtual` must appear before `eval` [ModifierMustAppearBefore]
  // CHECK:STDERR:   eval virtual fn VirtualEval();
  // CHECK:STDERR:        ^~~~~~~
  // CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:3: note: `eval` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   eval virtual fn VirtualEval();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  eval virtual fn VirtualEval();
}

// --- fail_not_fn.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+4]]:1: error: `eval` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: eval var x: i32;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval var x: i32;

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+4]]:1: error: `eval` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: eval class C {};
// CHECK:STDERR: ^~~~
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

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: eval fn NoneThenEval() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-10]]:1: note: previously not declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: fn NoneThenEval();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
eval fn NoneThenEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: musteval fn NoneThenMustEval() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-18]]:1: note: previously not declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: fn NoneThenMustEval();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
musteval fn NoneThenMustEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is not `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fn EvalThenNone() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-26]]:1: note: previously declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: eval fn EvalThenNone();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn EvalThenNone() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: musteval fn EvalThenMustEval() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-34]]:1: note: previously declared as `eval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: eval fn EvalThenMustEval();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
musteval fn EvalThenMustEval() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is not `musteval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: fn MustEvalThenNone() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-42]]:1: note: previously declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: musteval fn MustEvalThenNone();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MustEvalThenNone() {}

// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because new function is `eval` [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR: eval fn MustEvalThenEval() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_differ_in_redecl.carbon:[[@LINE-50]]:1: note: previously declared as `musteval` [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR: musteval fn MustEvalThenEval();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
eval fn MustEvalThenEval() {}
