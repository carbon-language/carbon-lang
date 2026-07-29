// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_guard_parens_only_parse_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_guard_parens_only_parse_errors.carbon

fn f() -> i32 {
  match (true) {
    // CHECK:STDERR: fail_missing_guard_parens_only_parse_errors.carbon:[[@LINE+4]]:33: error: expected `)` [ExpectedMatchCaseGuardCloseParen]
    // CHECK:STDERR:     case x: bool if (x == false => { return 1; }
    // CHECK:STDERR:                                 ^~
    // CHECK:STDERR:
    case x: bool if (x == false => { return 1; }
    case y: bool if (not y) => { return 2; }
    // CHECK:STDERR: fail_missing_guard_parens_only_parse_errors.carbon:[[@LINE+4]]:21: error: expected `(` after `if` [ExpectedMatchCaseGuardOpenParen]
    // CHECK:STDERR:     case z: bool if true or false) => { return 3; }
    // CHECK:STDERR:                     ^~~~
    // CHECK:STDERR:
    case z: bool if true or false) => { return 3; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_guard_parens_only_parse_errors.carbon
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
// CHECK:STDOUT:   │ │   │ ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │   ├─MatchCondition ')'
// CHECK:STDOUT:   │ │ ╭─MatchStatementStart '{'
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │     │ ╭─MatchCaseGuardIntroducer 'if'
// CHECK:STDOUT:   │ │ │     │ ├─MatchCaseGuardStart '('
// CHECK:STDOUT:   │ │ │     │ │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │     │ │ ├─BoolLiteralFalse 'false'
// CHECK:STDOUT:   │ │ │     │ ├─InfixOperatorEqualEqual '=='
// CHECK:STDOUT:   │ │ │     ├─MatchCaseGuard '=>' ❌
// CHECK:STDOUT:   │ │ │   ╭─MatchCase 'case' ❌
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart 'case' ❌
// CHECK:STDOUT:   │ │ ├─MatchHandler 'case' ❌
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │     │ ╭─MatchCaseGuardIntroducer 'if'
// CHECK:STDOUT:   │ │ │     │ ├─MatchCaseGuardStart 'true' ❌
// CHECK:STDOUT:   │ │ │     │ ├─InvalidParse 'true' ❌
// CHECK:STDOUT:   │ │ │     ├─MatchCaseGuard 'true' ❌
// CHECK:STDOUT:   │ │ │   ╭─MatchCase 'true' ❌
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart 'true' ❌
// CHECK:STDOUT:   │ │ ├─MatchHandler 'true' ❌
// CHECK:STDOUT:   │ ├─MatchStatement '}'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
