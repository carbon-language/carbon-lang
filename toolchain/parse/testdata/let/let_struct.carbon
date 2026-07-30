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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 17},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: let_struct_shorthand.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello shorthand"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: let_struct_mixed.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello mixed"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: let_ref_struct.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:             {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello ref"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 32},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: let_ref_struct_shorthand.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello ref shorthand"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
