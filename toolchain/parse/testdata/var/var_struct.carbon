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

var {t: str, j: 32} = {.t = "hello shorthand", .j = 1};

// CHECK:STDOUT: - filename: var_struct.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:               {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'StructPattern', text: '}', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 's'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'i'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 30},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: var_struct_shorthand.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 't'},
// CHECK:STDOUT:             {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'j'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPattern', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 't'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello shorthand"'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'j'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
