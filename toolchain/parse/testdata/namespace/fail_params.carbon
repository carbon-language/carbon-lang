// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:1: error: unexpected parameters in name declaration [UnexpectedParamsInDeclName]
// CHECK:STDERR: namespace A();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
namespace A();

// Parameters are ignored for error recovery.
fn A.F() {}

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:1: error: unexpected parameters in name declaration [UnexpectedParamsInDeclName]
// CHECK:STDERR: namespace B(n: i32);
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
namespace B(n: i32);

// CHECK:STDERR: fail_params.carbon:[[@LINE+3]]:1: error: unexpected parameters in name declaration [UnexpectedParamsInDeclName]
// CHECK:STDERR: namespace C[T:! type](x: T);
// CHECK:STDERR: ^~~~~~~~~
namespace C[T:! type](x: T);


// CHECK:STDOUT: - filename: fail_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:         {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
