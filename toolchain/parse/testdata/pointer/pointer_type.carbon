// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/pointer_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/pointer_type.carbon

fn F(p: i32*) -> i32* {
  f32* + 0;
  1 * bool* * 2;
  return 3 * i32* * *p;
}

var T: type = if true then i32* else f64*;

// CHECK:STDOUT: - filename: pointer_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'p'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │   ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   │ │   ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─IntLiteral '0'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │     │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │     ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─IntLiteral '3'
// CHECK:STDOUT:   │ │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │ │ ├─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │ ├─InfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │   ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ ╭─FloatTypeLiteral 'f64'
// CHECK:STDOUT:   │ │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─IfExprElse 'else'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
