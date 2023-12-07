// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// The following should work, but the support for these has not been
// implemented yet.
interface Foo {
  // CHECK:STDERR: associated_constants.carbon:[[@LINE+3]]:15: ERROR: Expected `=`; `let` declaration must have an initializer.
  // CHECK:STDERR:   let T:! type;
  // CHECK:STDERR:               ^
  let T:! type;
  // CHECK:STDERR: associated_constants.carbon:[[@LINE+3]]:3: ERROR: Unrecognized declaration introducer.
  // CHECK:STDERR:   final let I: i32 = 4;
  // CHECK:STDERR:   ^
  final let I: i32 = 4;
  // CHECK:STDERR: associated_constants.carbon:[[@LINE+3]]:3: ERROR: Unrecognized declaration introducer.
  // CHECK:STDERR:   default let D: bool = true;
  // CHECK:STDERR:   ^
  default let D: bool = true;
}

// CHECK:STDOUT: - filename: associated_constants.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'Name', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'Name', text: 'T'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'type'},
// CHECK:STDOUT:         {kind: 'GenericPatternBinding', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'EmptyDecl', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'EmptyDecl', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
