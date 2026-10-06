// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if_expr/precedence.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if_expr/precedence.carbon

fn F(b: bool) -> bool {
  return if b and b then b and b else b or b;
}

// CHECK:STDOUT: - filename: precedence.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │       ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │     ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │ │     ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │   ╭─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │   ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─ShortCircuitOperandOr 'or'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ ├─ShortCircuitOperatorOr 'or'
// CHECK:STDOUT:   │ │ ├─IfExprElse 'else'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
