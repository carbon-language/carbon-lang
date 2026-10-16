// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_mix_value_and_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_mix_value_and_type.carbon

// CHECK:STDERR: fail_mix_value_and_type.carbon:[[@LINE+2]]:17: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: {.a = 0, b: i32} = {};

// CHECK:STDOUT: - filename: fail_mix_value_and_type.carbon
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
// CHECK:STDOUT:   │ │   │ ├─InvalidParse 'b' has_error
// CHECK:STDOUT:   │ │   ├─StructLiteral '}' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
