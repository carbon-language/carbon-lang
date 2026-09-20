// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_comma_repeat_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_comma_repeat_in_type.carbon

// CHECK:STDERR: fail_comma_repeat_in_type.carbon:[[@LINE+2]]:17: error: expected `.field: field_type` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: {.a: i32,,} = {};

// CHECK:STDOUT: - filename: fail_comma_repeat_in_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─StructTypeLiteralStart '{'
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │   │ ├─StructTypeLiteralComma ','
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   │ ├─StructTypeLiteralComma ','
// CHECK:STDOUT:   │ │   ├─StructTypeLiteral '}' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
