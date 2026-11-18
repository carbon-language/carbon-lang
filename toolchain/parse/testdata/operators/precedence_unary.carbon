// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_unary.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_unary.carbon

fn F(p: i32*) {
  // Numeric, bitwise, and bit-shift binary operators are lower precedence than
  // numeric and bitwise unary operators, which are lower precedence than
  // dereference.
  -*p + ^*p;
  -*p | ^*p;
  -*p << ^*p;
}

// CHECK:STDOUT: - filename: precedence_unary.carbon
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
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │     ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   │ ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorCaret '^'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │     ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   │ ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorCaret '^'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPipe '|'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │     ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   │ ╭─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorCaret '^'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorLessLess '<<'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
