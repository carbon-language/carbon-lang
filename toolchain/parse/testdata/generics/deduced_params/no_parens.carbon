// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/deduced_params/no_parens.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/deduced_params/no_parens.carbon

// --- class.carbon

class Foo[];

class Foo[a: i32];

// --- interface.carbon

interface Bar[] {}

interface Bar[a: i32] {}

// --- qualified.carbon

interface Bar[].Baz[] {}

// CHECK:STDOUT: - filename: class.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: interface.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: qualified.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameQualifierWithParams', text: '.', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Baz'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
