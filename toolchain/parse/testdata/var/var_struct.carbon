// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/var_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/var_struct.carbon

// --- var_struct.carbon

var {.s = s: str, .i = i: 32} = {.s = "hello", .i = 0};

// --- var_struct_shorthand.carbon

var {s: str, i: 32} = {.s = "hello shorthand", .i = 1};

// --- var_struct_mixed.carbon

var {.s = s: str, i: 32} = {.s = "hello mixed", .i = 1};

// CHECK:STDOUT: - filename: var_struct.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │   │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │   │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IntLiteral '32'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ╭─StructPattern '}'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: var_struct_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ╭─StructPattern '}'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello shorthand"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: var_struct_mixed.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │   │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ╭─StructPattern '}'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello mixed"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
