// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/pointer_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/pointer_value.carbon

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;
  var q: i32** = &p;
  return **q;
}

// CHECK:STDOUT: - filename: pointer_value.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'p'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ├─PrefixOperatorAmp '&'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'q'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │ ├─PrefixOperatorAmp '&'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'q'
// CHECK:STDOUT:   │ │ │ ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │ ├─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
