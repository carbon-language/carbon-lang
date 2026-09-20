// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_comma_repeat_in_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_comma_repeat_in_value.carbon

// CHECK:STDERR: fail_comma_repeat_in_value.carbon:[[@LINE+2]]:16: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: {.a = 0,,} = {};

// CHECK:STDOUT: - filename: fail_comma_repeat_in_value.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ │   │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │   │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │   ├─StructLiteral '}' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
