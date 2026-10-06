// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/one_entry_no_comma.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/one_entry_no_comma.carbon

var z: {.n: i32} = {.n = 4};

// CHECK:STDOUT: - filename: one_entry_no_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─StructTypeLiteralStart '{'
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │   ├─StructTypeLiteral '}'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '4'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
