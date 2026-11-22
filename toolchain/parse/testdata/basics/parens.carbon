// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/parens.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/parens.carbon

fn F(n: i32) -> i32 {
  return ((((n))));
}

// CHECK:STDOUT: - filename: parens.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ │ ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ │ │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
