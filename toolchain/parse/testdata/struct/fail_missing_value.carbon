// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_missing_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_missing_value.carbon

// CHECK:STDERR: fail_missing_value.carbon:[[@LINE+2]]:12: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
var x: {.a=} = {};

// CHECK:STDOUT: - filename: fail_missing_value.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ ├─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │   ├─StructLiteral '}' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
