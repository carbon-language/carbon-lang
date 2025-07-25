// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_associated_const_default_initializer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_associated_const_default_initializer.carbon

interface MagicBox {
  // CHECK:STDERR: fail_associated_const_default_initializer.carbon:[[@LINE+4]]:30: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let secretMessage:! String = "This is a super secret";
  // CHECK:STDERR:                              ^
  // CHECK:STDERR:
  let secretMessage:! String = "This is a super secret";
} 

// CHECK:STDOUT: - filename: fail_associated_const_default_initializer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'MagicBox'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'secretMessage'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantNameAndType', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
