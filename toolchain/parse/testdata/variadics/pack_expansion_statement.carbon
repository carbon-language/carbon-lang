// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/variadics/pack_expansion_statement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/variadics/pack_expansion_statement.carbon

// --- expr_statement.carbon

fn F() {
  ... G(expand x);
}

// --- var_statement.carbon

fn F() {
  ... var y: auto = expand x;
}

// --- compound_statements.carbon

fn F() {
  ... if (expand c) {
    G(expand x);
  } else {
    H(expand y);
  }
  ... while (expand c) {}
  ... return;
}

// --- nested.carbon

fn F() {
  ... ... G(expand x);
}

// --- fail_missing_statement.carbon

fn F() {
  ...
// CHECK:STDERR: fail_missing_statement.carbon:[[@LINE+4]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// --- fail_missing_semi.carbon

fn F() {
  ... G()
// CHECK:STDERR: fail_missing_semi.carbon:[[@LINE+4]]:1: error: expected `;` after expression statement [ExpectedExprSemi]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// CHECK:STDOUT: - filename: expr_statement.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: var_statement.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: compound_statements.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │ │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ │   │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │     ╭─IdentifierNameExpr 'H'
// CHECK:STDOUT:   │ │ │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ │   │ ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │ │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │   ╭─WhileConditionStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ ╭─WhileCondition ')'
// CHECK:STDOUT:   │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ ├─WhileStatement 'while'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │   │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ │   ├─PrefixOperatorExpand 'expand'
// CHECK:STDOUT:   │ │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_statement.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │ ╭─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ ├─ExprStatement '...' has_error
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackExpansionStart '...'
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─ExprStatement ')' has_error
// CHECK:STDOUT:   │ ├─PackExpansionStatement '...'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
