// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_precedence_and_or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_precedence_and_or.carbon

fn F() {
  // CHECK:STDERR: fail_precedence_and_or.carbon:[[@LINE+2]]:11-12: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:
  a and b or c;
}

// CHECK:STDOUT: - filename: fail_precedence_and_or.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │         ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │       ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │     ╭─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ │   ╭─ShortCircuitOperandOr 'or' has_error
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ ╭─ShortCircuitOperatorOr 'or' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
