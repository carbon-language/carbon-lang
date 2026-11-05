// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/unexpected_runtime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/unexpected_runtime.carbon

// --- fail_unexpected_runtime_base.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {
  let T: type;
}

class D {
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_base.carbon:[[@LINE+7]]:18
  // CHECK:STDERR: ->     │   extend impl as x where .T = () {}
  // CHECK:STDERR:        ·                  ┳
  // CHECK:STDERR:        ·                  ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                     ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend impl as x where .T = () {}
}

// --- fail_unexpected_runtime_rewrite_lhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {
  let T: type;
}

class D {
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_rewrite_lhs.carbon:[[@LINE+16]]:18
  // CHECK:STDERR: ->     │   extend impl as Z where x = () {}
  // CHECK:STDERR:        ·                  ┳━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━┫ error: semantics TODO:
  // CHECK:STDERR:        ·                     ┃        `handle invalid parse trees in `check``
  // CHECK:STDERR:        ·                     ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_rewrite_lhs.carbon:[[@LINE+8]]:28
  // CHECK:STDERR: ->     │   extend impl as Z where x = () {}
  // CHECK:STDERR:        ·                            ┳
  // CHECK:STDERR:        ·                            ┗━━┫ error: requirement can only use `=` after
  // CHECK:STDERR:        ·                               ┃        `.member` designator
  // CHECK:STDERR:        ·                               ┃        [RequirementEqualAfterNonDesignator]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend impl as Z where x = () {}
}

// --- fail_unexpected_runtime_rewrite_rhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {
  let T: type;
}

class D {
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_rewrite_rhs.carbon:[[@LINE+8]]:31
  // CHECK:STDERR: ->     │   extend impl as Z where .T = x {}
  // CHECK:STDERR:        ·                               ┳
  // CHECK:STDERR:        ·                               ┗━━┫ error: expression is runtime; expected
  // CHECK:STDERR:        ·                                  ┃        constant
  // CHECK:STDERR:        ·                                  ┃        [EvalRequiresConstantValue]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend impl as Z where .T = x {}
}

// --- fail_unexpected_runtime_impls_lhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z(T: type) {}

class D {
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_impls_lhs.carbon:[[@LINE+7]]:29
  // CHECK:STDERR: ->     │   extend impl as type where x impls Z(.Self) {}
  // CHECK:STDERR:        ·                             ┳
  // CHECK:STDERR:        ·                             ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                                ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend impl as type where x impls Z(.Self) {}
}

// --- fail_unexpected_runtime_impls_rhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {}

class D {
  // CHECK:STDERR:        ╭─┤ fail_unexpected_runtime_impls_rhs.carbon:[[@LINE+8]]:41
  // CHECK:STDERR: ->     │   extend impl as type where .Self impls x {}
  // CHECK:STDERR:        ·                                         ┳
  // CHECK:STDERR:        ·                                         ┗━━┫ error: cannot evaluate type
  // CHECK:STDERR:        ·                                            ┃        expression
  // CHECK:STDERR:        ·                                            ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend impl as type where .Self impls x {}
}
