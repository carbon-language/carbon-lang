// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/deduced_params/fail_no_parens.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/deduced_params/fail_no_parens.carbon

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+4]]:12: error: a `(` for parameters is required after implicit parameters [ParamsRequiredAfterImplicit]
// CHECK:STDERR: class Foo[];
// CHECK:STDERR:            ^
// CHECK:STDERR:
class Foo[];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+4]]:18: error: a `(` for parameters is required after implicit parameters [ParamsRequiredAfterImplicit]
// CHECK:STDERR: class Foo[a: i32];
// CHECK:STDERR:                  ^
// CHECK:STDERR:
class Foo[a: i32];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+4]]:17: error: a `(` for parameters is required after implicit parameters [ParamsRequiredAfterImplicit]
// CHECK:STDERR: interface Bar[] {}
// CHECK:STDERR:                 ^
// CHECK:STDERR:
interface Bar[] {}

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+4]]:23: error: a `(` for parameters is required after implicit parameters [ParamsRequiredAfterImplicit]
// CHECK:STDERR: interface Bar[a: i32] {}
// CHECK:STDERR:                       ^
// CHECK:STDERR:
interface Bar[a: i32] {}

// CHECK:STDOUT: - filename: fail_no_parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
