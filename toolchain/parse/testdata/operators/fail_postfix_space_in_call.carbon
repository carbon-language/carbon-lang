// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_postfix_space_in_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_postfix_space_in_call.carbon

// CHECK:STDERR: fail_postfix_space_in_call.carbon:[[@LINE+2]]:18: error: whitespace is not allowed before this unary operator [UnaryOperatorHasWhitespace]
// CHECK:STDERR:
var n: i8 = F(i8 *);

// CHECK:STDOUT: - filename: fail_postfix_space_in_call.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i8'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'F'
// CHECK:STDOUT:   │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ ╭─IntTypeLiteral 'i8'
// CHECK:STDOUT:   │ │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─CallExpr ')'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
