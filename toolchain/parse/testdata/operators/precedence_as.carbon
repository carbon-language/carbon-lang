// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_as.carbon

fn F(n: i32) {
  // Type operators and unary operators are higher precedence than `as`.
  -n as const i32;
  &n as i32*;
  ^n as i32;
  *n as i32;
}

// CHECK:STDOUT: - filename: precedence_as.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorAmp '&'
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorCaret '^'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
