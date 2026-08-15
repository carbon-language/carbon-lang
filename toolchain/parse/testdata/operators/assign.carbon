// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/assign.carbon

fn F() {
  var a: i32 = 0;
  var b: i32 = 1;
  a = b;
  a *= b;
  a /= b;
  a += b;
  a -= b;
  a %= b;
  a &= b;
  a |= b;
  a ^= b;
  a <<= b;
  a >>= b;
  ++a;
  --a;
}

// CHECK:STDOUT: - filename: assign.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqual '='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStarEqual '*='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorSlashEqual '/='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlusEqual '+='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorMinusEqual '-='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPercentEqual '%='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAmpEqual '&='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPipeEqual '|='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorCaretEqual '^='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorLessLessEqual '<<='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorGreaterGreaterEqual '>>='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─PrefixOperatorPlusPlus '++'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─PrefixOperatorMinusMinus '--'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
