// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/fail_bracket_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/fail_bracket_recovery.carbon

// This is a valid parse tree even though the lex errors.
fn F() {
  var a: array(i32, 3) = (0, 1, 2);
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+3]]:5: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+2]]:7: note: possibly missing `]` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  (a[1) + 1;
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+9]]:15: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+8]]:16: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+6]]:26: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+5]]:28: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+3]]:27: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+2]]:28: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var x: {.y: (} = {.y = ((};
}

// CHECK:STDOUT: - filename: fail_bracket_recovery.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │ │   │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ │   ├─ArrayExpr ')'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │     ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │     │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     │ ╭─IndexExprStart '['
// CHECK:STDOUT:   │ │     │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │     ├─IndexExpr ']'
// CHECK:STDOUT:   │ │   ╭─ParenExpr ')'
// CHECK:STDOUT:   │ │   ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─StructTypeLiteralStart '{'
// CHECK:STDOUT:   │ │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │   │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │ │   ├─StructTypeLiteral '}'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteral '}'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
