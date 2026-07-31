// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/infix.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/infix.carbon

fn F() {
  n * n;
  n ^ n;
  n >= n;
  n >> n;
  n / n;
}

// CHECK:STDOUT: - filename: infix.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorCaret '^'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorGreaterEqual '>='
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorGreaterGreater '>>'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorSlash '/'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
