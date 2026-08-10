// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_case_arrow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_case_arrow.carbon

fn f() -> i32 {
  match (3) {
    // CHECK:STDERR: fail_missing_case_arrow.carbon:[[@LINE+4]]:17: error: expected `=>` introducing statement block [ExpectedMatchCaseArrow]
    // CHECK:STDERR:     case x: i32 { return 2; }
    // CHECK:STDERR:                 ^
    // CHECK:STDERR:
    case x: i32 { return 2; }
    // CHECK:STDERR: fail_missing_case_arrow.carbon:[[@LINE+4]]:29: error: expected `=>` introducing statement block [ExpectedMatchCaseArrow]
    // CHECK:STDERR:     case x: i32 if (x == 3) { return 3; }
    // CHECK:STDERR:                             ^
    // CHECK:STDERR:
    case x: i32 if (x == 3) { return 3; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_case_arrow.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'f'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─MatchIntroducer 'match'
// CHECK:STDOUT:   │ │   │ ╭─MatchConditionStart '('
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │   ├─MatchCondition ')'
// CHECK:STDOUT:   │ │ ╭─MatchStatementStart '{'
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │   ╭─MatchCase '{' has_error
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '{' has_error
// CHECK:STDOUT:   │ │ ├─MatchHandler '{' has_error
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │     │ ╭─MatchCaseGuardIntroducer 'if'
// CHECK:STDOUT:   │ │ │     │ ├─MatchCaseGuardStart '('
// CHECK:STDOUT:   │ │ │     │ │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │     │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ │     │ ├─InfixOperatorEqualEqual '=='
// CHECK:STDOUT:   │ │ │     ├─MatchCaseGuard ')'
// CHECK:STDOUT:   │ │ │   ╭─MatchCase '{' has_error
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '{' has_error
// CHECK:STDOUT:   │ │ ├─MatchHandler '{' has_error
// CHECK:STDOUT:   │ ├─MatchStatement '}'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
