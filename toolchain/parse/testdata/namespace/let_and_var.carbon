// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/let_and_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/let_and_var.carbon

var NS.va: type;
var NS.vb: i32 = 42;

let NS.la: type;
let NS.lb: i32 = 42;

var NS.Sub.a: i32;
var NS.Sub.b: type = i32;

let NS.Sub.a: i32;
let NS.Sub.b: type = i32;

var NS._: i32;
let NS._: i32;

// CHECK:STDOUT: - filename: let_and_var.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'va'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'vb'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '42'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'la'},
// CHECK:STDOUT:         {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'lb'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '42'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:         {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
