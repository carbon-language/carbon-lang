// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/associative.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/associative.carbon

fn F() {
  a and b and c;
}

// CHECK:STDOUT: - filename: associative.carbon
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
// CHECK:STDOUT:   │ │   ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ ╭─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
