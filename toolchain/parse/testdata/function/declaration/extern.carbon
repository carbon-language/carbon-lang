// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration/extern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration/extern.carbon

// --- extern.carbon

extern fn G();

// --- library.carbon

extern library "foo" fn G();

// --- default_library.carbon

extern library default fn G();

// --- fail_name_missing.carbon

// CHECK:STDERR: fail_name_missing.carbon:[[@LINE+4]]:16: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR: extern library fn G();
// CHECK:STDERR:                ^~
// CHECK:STDERR:
extern library fn G();

// --- fail_identifier.carbon

// CHECK:STDERR: fail_identifier.carbon:[[@LINE+8]]:16: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR: extern library foo fn G();
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_identifier.carbon:[[@LINE+4]]:16: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: extern library foo fn G();
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
extern library foo fn G();

// CHECK:STDOUT: - filename: extern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'ExternModifier', text: 'extern'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'LibraryName', text: '"foo"'},
// CHECK:STDOUT:         {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExternModifierWithLibrary', text: 'extern', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: default_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'DefaultLibrary', text: 'default'},
// CHECK:STDOUT:         {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExternModifierWithLibrary', text: 'extern', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_name_missing.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'LibraryName', text: 'fn', has_error: yes},
// CHECK:STDOUT:         {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExternModifierWithLibrary', text: 'extern', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'foo', has_error: yes},
// CHECK:STDOUT:           {kind: 'LibraryName', text: 'foo', has_error: yes},
// CHECK:STDOUT:         {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExternModifierWithLibrary', text: 'extern', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
