// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if_expr/in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if_expr/in_type.carbon

var n: i32;

fn F() -> if true then i32 else i32* {
  return if true then n else &n;
}

// CHECK:STDOUT: - filename: in_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │     ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │   │   ╭─IfExprIf 'if'
// CHECK:STDOUT:   │   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │   ├─IfExprThen 'then'
// CHECK:STDOUT:   │   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   │ ╭─IfExprElse 'else'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ │ ├─PrefixOperatorAmp '&'
// CHECK:STDOUT:   │ │ ├─IfExprElse 'else'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
