// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_in_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_in_interface.carbon

interface I {
  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   let var a: ();
  // CHECK:STDERR:       ^~~
  // CHECK:STDERR:
  let var a: ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   var b: ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  var b: ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   let var c:! ();
  // CHECK:STDERR:       ^~~
  // CHECK:STDERR:
  let var c:! ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   var d:! ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  var d:! ();
}

// CHECK:STDOUT: - filename: fail_in_interface.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParseStart', text: 'var', has_error: yes},
// CHECK:STDOUT:       {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParseStart', text: 'var', has_error: yes},
// CHECK:STDOUT:       {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
