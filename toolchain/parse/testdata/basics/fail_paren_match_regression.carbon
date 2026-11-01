// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/fail_paren_match_regression.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/fail_paren_match_regression.carbon

// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+6]]:4: error: expected pattern [ExpectedPattern]
// CHECK:STDERR:
// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+4]]:11: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
// CHECK:STDERR:
// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+2]]:15: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
var = (foo {})

// CHECK:STDOUT: - filename: fail_paren_match_regression.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ ╭─InvalidParse '=' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ ├─ParenExpr ')' has_error
// CHECK:STDOUT:   ├─VariableDecl ')' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
