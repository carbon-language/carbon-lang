// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/let_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/let_struct.carbon

// --- let_struct.carbon

let {.s = s: str, .i = i: i32} = {.s = "hello", .i = 1};

// --- let_struct_shorthand.carbon

let {s: str, i: i32} = {.s = "hello shorthand", .i = 2};

// --- let_struct_mixed.carbon

let {.s = s: str, i: i32} = {.s = "hello mixed", .i = 2};

// --- let_ref_struct.carbon

let {.s = ref s: str, .i = i: i32} = {.s = "hello ref", .i = 3};

// --- let_ref_struct_shorthand.carbon

let {ref s: str, i: i32} = {.s = "hello ref shorthand", .i = 4};

// CHECK:STDOUT: - filename: let_struct.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: let_struct_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello shorthand"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: let_struct_mixed.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello mixed"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: let_ref_struct.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello ref"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: let_ref_struct_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello ref shorthand"'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '4'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
