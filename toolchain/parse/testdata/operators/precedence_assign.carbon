// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_assign.carbon

fn F(c: bool) {
  var a: i32;
  var b: i32;
  var p: i32*;
  *p = if c then 1 else 2;
  // This is valid to _parse_ even though rejected semantically.
  (if c then a else b) += if c then 1 else 2;
}

// CHECK:STDOUT: - filename: precedence_assign.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'p'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │   │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │   │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │   ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqual '='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │     │   ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │     │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │     │ │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │     │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │     ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │   ╭─ParenExpr ')'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │   │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │   │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │   ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlusEqual '+='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
