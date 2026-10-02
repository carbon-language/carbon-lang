// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_mix_with_unknown.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_mix_with_unknown.carbon

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+4]]:25: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR:
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+2]]:29: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: i32 = {.a = 1, .b, .c: i32};

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+4]]:26: error: expected `.field: field_type` [ExpectedStructLiteralField]
// CHECK:STDERR:
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+2]]:31: error: expected `.field: field_type` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: i32 = {.a: i32, .b, .c = 1};

// CHECK:STDOUT: - filename: fail_mix_with_unknown.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ ├─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ ├─InvalidParse '.' has_error
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │ ├─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ ├─InvalidParse '.' has_error
// CHECK:STDOUT:   │ ├─StructLiteral '}' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructTypeLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │ ├─StructTypeLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ ├─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ ├─InvalidParse '.' has_error
// CHECK:STDOUT:   │ │ ├─StructTypeLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │ ├─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ ├─InvalidParse '.' has_error
// CHECK:STDOUT:   │ ├─StructTypeLiteral '}' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
