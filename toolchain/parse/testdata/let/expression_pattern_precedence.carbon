// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/expression_pattern_precedence.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/expression_pattern_precedence.carbon

// --- parens.carbon

fn F() {
  let (a == b) = c;

  let (a and b) = c;

  let (if true then 1 else 2) = 3;
}

// --- fail_relational.carbon

fn G() {
  // CHECK:STDERR: fail_relational.carbon:[[@LINE+4]]:9: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let a == b = c;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  let a == b = c;
}

// --- fail_logical.carbon

fn H() {
  // CHECK:STDERR: fail_logical.carbon:[[@LINE+4]]:9: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let a and b = c;
  // CHECK:STDERR:         ^~~
  // CHECK:STDERR:
  let a and b = c;
}

// --- fail_if.carbon

fn I() {
  // CHECK:STDERR: fail_if.carbon:[[@LINE+4]]:7: error: parentheses are required around this unary `if` operator [UnaryOperatorRequiresParentheses]
  // CHECK:STDERR:   let if true then 1 else 2 = 3;
  // CHECK:STDERR:       ^~
  // CHECK:STDERR:
  let if true then 1 else 2 = 3;
}

// CHECK:STDOUT: - filename: parens.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ ├─InfixOperatorEqualEqual '=='
// CHECK:STDOUT:   │ │ ├─ParenPattern ')'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ │ │ ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ ├─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ │ ├─ParenPattern ')'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ │   ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ │ ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ├─ParenPattern ')'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_relational.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ ├─LetDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_logical.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ ├─LetDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_if.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'I'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │   ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
