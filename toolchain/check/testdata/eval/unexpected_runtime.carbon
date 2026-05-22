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
  let T:! type;
}

class D {
  // CHECK:STDERR: fail_unexpected_runtime_base.carbon:[[@LINE+4]]:18: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   extend impl as x where .T = () {}
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  extend impl as x where .T = () {}
}

// --- fail_unexpected_runtime_rewrite_lhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {
  let T:! type;
}

class D {
  // CHECK:STDERR: fail_unexpected_runtime_rewrite_lhs.carbon:[[@LINE+8]]:18: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   extend impl as Z where x = () {}
  // CHECK:STDERR:                  ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unexpected_runtime_rewrite_lhs.carbon:[[@LINE+4]]:28: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
  // CHECK:STDERR:   extend impl as Z where x = () {}
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  extend impl as Z where x = () {}
}

// --- fail_unexpected_runtime_rewrite_rhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {
  let T:! type;
}

class D {
  // CHECK:STDERR: fail_unexpected_runtime_rewrite_rhs.carbon:[[@LINE+4]]:31: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  // CHECK:STDERR:   extend impl as Z where .T = x {}
  // CHECK:STDERR:                               ^
  // CHECK:STDERR:
  extend impl as Z where .T = x {}
}

// --- fail_unexpected_runtime_impls_lhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {}

class D {
  // CHECK:STDERR: fail_unexpected_runtime_impls_lhs.carbon:[[@LINE+4]]:29: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   extend impl as type where x impls Z {}
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  extend impl as type where x impls Z {}
}

// --- fail_unexpected_runtime_impls_rhs.carbon
library "[[@TEST_NAME]]";

var x: type;

interface Z {}

class D {
  // CHECK:STDERR: fail_unexpected_runtime_impls_rhs.carbon:[[@LINE+4]]:41: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   extend impl as type where .Self impls x {}
  // CHECK:STDERR:                                         ^
  // CHECK:STDERR:
  extend impl as type where .Self impls x {}
}
