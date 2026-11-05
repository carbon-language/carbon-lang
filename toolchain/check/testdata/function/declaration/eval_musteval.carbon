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

// CHECK:STDERR:        ╭─┤ fail_conflict.carbon:[[@LINE+8]]:6
// CHECK:STDERR: ->     │ eval musteval fn Mixed();
// CHECK:STDERR:        · ──┬─ ━━━━┳━━━
// CHECK:STDERR:        ·   │      ┗━━┫ error: `musteval` not allowed on declaration with `eval`
// CHECK:STDERR:        ·   │         ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·   ╰───────────┤ `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval musteval fn Mixed();

// --- fail_order.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_order.carbon:[[@LINE+8]]:6
// CHECK:STDERR: ->     │ eval export fn ExportEval();
// CHECK:STDERR:        · ──┬─ ━━━┳━━
// CHECK:STDERR:        ·   │     ┗━━┫ error: `export` must appear before `eval`
// CHECK:STDERR:        ·   │        ┃        [ModifierMustAppearBefore]
// CHECK:STDERR:        ·   ╰──────────┤ `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval export fn ExportEval();

// CHECK:STDERR:        ╭─┤ fail_order.carbon:[[@LINE+8]]:6
// CHECK:STDERR: ->     │ eval private fn PrivateEval();
// CHECK:STDERR:        · ──┬─ ━━━┳━━━
// CHECK:STDERR:        ·   │     ┗━━┫ error: `private` must appear before `eval`
// CHECK:STDERR:        ·   │        ┃        [ModifierMustAppearBefore]
// CHECK:STDERR:        ·   ╰──────────┤ `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval private fn PrivateEval();

class C {
  // CHECK:STDERR:        ╭─┤ fail_order.carbon:[[@LINE+8]]:8
  // CHECK:STDERR: ->     │   eval virtual fn VirtualEval();
  // CHECK:STDERR:        ·   ──┬─ ━━━┳━━━
  // CHECK:STDERR:        ·     │     ┗━━┫ error: `virtual` must appear before `eval`
  // CHECK:STDERR:        ·     │        ┃        [ModifierMustAppearBefore]
  // CHECK:STDERR:        ·     ╰──────────┤ `eval` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  eval virtual fn VirtualEval();
}

// --- fail_not_fn.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_not_fn.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ eval var x: i32;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `eval` not allowed on `var` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval var x: i32;

// CHECK:STDERR:        ╭─┤ fail_not_fn.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ eval class C {};
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `eval` not allowed on `class` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
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

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ fn NoneThenEval();
// CHECK:STDERR:        · ─────────┬────────
// CHECK:STDERR:        ·          ╰──┤ previously not declared as `eval`
// CHECK:STDERR:        ·             │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ eval fn NoneThenEval() {}
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: function redeclaration differs because new function is `eval`
// CHECK:STDERR:        ·           ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval fn NoneThenEval() {}

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ fn NoneThenMustEval();
// CHECK:STDERR:        · ───────────┬──────────
// CHECK:STDERR:        ·            ╰──┤ previously not declared as `musteval`
// CHECK:STDERR:        ·               │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ musteval fn NoneThenMustEval() {}
// CHECK:STDERR:        · ━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·    ┗━━┫ error: function redeclaration differs because new function is `musteval`
// CHECK:STDERR:        ·       ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
musteval fn NoneThenMustEval() {}

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ eval fn EvalThenNone();
// CHECK:STDERR:        · ───────────┬───────────
// CHECK:STDERR:        ·            ╰──┤ previously declared as `eval`
// CHECK:STDERR:        ·               │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn EvalThenNone() {}
// CHECK:STDERR:        · ━━━┳━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·    ┗━━┫ error: function redeclaration differs because new function is not `eval`
// CHECK:STDERR:        ·       ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn EvalThenNone() {}

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ eval fn EvalThenMustEval();
// CHECK:STDERR:        · ─────────────┬─────────────
// CHECK:STDERR:        ·              ╰──┤ previously declared as `eval`
// CHECK:STDERR:        ·                 │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ musteval fn EvalThenMustEval() {}
// CHECK:STDERR:        · ━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·    ┗━━┫ error: function redeclaration differs because new function is `musteval`
// CHECK:STDERR:        ·       ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
musteval fn EvalThenMustEval() {}

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+13]]:1
// CHECK:STDERR:        │ musteval fn MustEvalThenNone();
// CHECK:STDERR:        · ───────────────┬───────────────
// CHECK:STDERR:        ·                ╰──┤ previously declared as `musteval`
// CHECK:STDERR:        ·                   │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn MustEvalThenNone() {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: function redeclaration differs because new function is not
// CHECK:STDERR:        ·    ┃        `musteval`
// CHECK:STDERR:        ·    ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn MustEvalThenNone() {}

// CHECK:STDERR:        ╭─┤ fail_differ_in_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ musteval fn MustEvalThenEval();
// CHECK:STDERR:        · ───────────────┬───────────────
// CHECK:STDERR:        ·                ╰──┤ previously declared as `musteval`
// CHECK:STDERR:        ·                   │ [FunctionRedeclEvaluationModePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ eval fn MustEvalThenEval() {}
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: function redeclaration differs because new function is `eval`
// CHECK:STDERR:        ·           ┃        [FunctionRedeclEvaluationModeDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
eval fn MustEvalThenEval() {}
