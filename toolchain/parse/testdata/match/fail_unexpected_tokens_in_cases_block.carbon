// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_unexpected_tokens_in_cases_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_unexpected_tokens_in_cases_block.carbon

fn f() -> i32 {
  var x: i32 = 3;
  match (x) {
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5-7: error: unexpected `let`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    let u: i32 =
    case y: i32 if (y > 2) => { return 1; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5-6: error: unexpected `if`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    if (true) {
      return 2;
    }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5-7: error: unexpected `var`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    var w
    case z: i32 if (z == 3) => { return 3; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5: error: unexpected `{`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    {
      break;
    }
    default => { return 3; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5-10: error: unexpected `return`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    return 1;
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+2]]:5: error: unexpected `(`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:
    ((), (), ())
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_unexpected_tokens_in_cases_block.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'f'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │   ╭─MatchIntroducer 'match'
// CHECK:STDOUT:   │ │   │ ╭─MatchConditionStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─MatchCondition ')'
// CHECK:STDOUT:   │ │ ╭─MatchStatementStart '{'
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │     │ ╭─MatchCaseGuardIntroducer 'if'
// CHECK:STDOUT:   │ │ │     │ ├─MatchCaseGuardStart '('
// CHECK:STDOUT:   │ │ │     │ │ ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │ │     │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ │     │ ├─InfixOperatorGreater '>'
// CHECK:STDOUT:   │ │ │     ├─MatchCaseGuard ')'
// CHECK:STDOUT:   │ │ │   ╭─MatchCase '=>'
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ │ ├─MatchHandler '}'
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │ ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │     │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │     │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │     ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │     │ ╭─MatchCaseGuardIntroducer 'if'
// CHECK:STDOUT:   │ │ │     │ ├─MatchCaseGuardStart '('
// CHECK:STDOUT:   │ │ │     │ │ ╭─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │ │     │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ │     │ ├─InfixOperatorEqualEqual '=='
// CHECK:STDOUT:   │ │ │     ├─MatchCaseGuard ')'
// CHECK:STDOUT:   │ │ │   ╭─MatchCase '=>'
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ │ ├─MatchHandler '}'
// CHECK:STDOUT:   │ │ │     ╭─MatchDefaultIntroducer 'default'
// CHECK:STDOUT:   │ │ │   ╭─MatchDefault '=>'
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ │ ├─MatchHandler '}'
// CHECK:STDOUT:   │ ├─MatchStatement '}' has_error
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
