// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/two_entries.carbon

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};

// CHECK:STDOUT: - filename: two_entries.carbon
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
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │   ├─StructTypeLiteral '}'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
