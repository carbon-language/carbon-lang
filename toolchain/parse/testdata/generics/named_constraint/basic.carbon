// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/basic.carbon

constraint Foo {
  // TODO: Add a require statement.
}

constraint ForwardDeclared;

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'NamedConstraintDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'NamedConstraintDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'ForwardDeclared'},
// CHECK:STDOUT:     {kind: 'NamedConstraintDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
