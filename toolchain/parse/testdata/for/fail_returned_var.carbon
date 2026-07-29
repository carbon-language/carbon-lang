// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/fail_returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/fail_returned_var.carbon

fn foo() -> i32 {
  // CHECK:STDERR: fail_returned_var.carbon:[[@LINE+4]]:8: error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:   for (returned var x: i32 in y) {
  // CHECK:STDERR:        ^~~~~~~~
  // CHECK:STDERR:
  for (returned var x: i32 in y) {
    return var;
  }
}

// CHECK:STDOUT: - filename: fail_returned_var.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ForHeaderStart '('
// CHECK:STDOUT:   │ │   │ ╭─InvalidParse 'returned' ❌
// CHECK:STDOUT:   │ │   ├─ForIn '(' ❌
// CHECK:STDOUT:   │ │ ╭─ForHeader ')' ❌
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ├─ReturnVarModifier 'var'
// CHECK:STDOUT:   │ │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─ForStatement 'for'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
