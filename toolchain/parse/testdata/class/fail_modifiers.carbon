// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_virtual_class.carbon

virtual class B
// CHECK:STDERR: fail_virtual_class.carbon:[[@LINE+4]]:1: ERROR: `virtual` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_impl_class.carbon

// CHECK:STDERR: fail_impl_class.carbon:[[@LINE+4]]:1: ERROR: `impl` introducer should be followed by a name.
// CHECK:STDERR: impl class
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl class

// --- fail_extend_adapt_base.carbon

class C {
  // CHECK:STDERR: fail_extend_adapt_base.carbon:[[@LINE+3]]:16: ERROR: Expected expression.
  // CHECK:STDERR:   extend adapt base B;
  // CHECK:STDERR:                ^~~~
  extend adapt base B;
}

// CHECK:STDOUT: - filename: fail_virtual_class.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: 'B', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_impl_class.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: 'class', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_extend_adapt_base.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AdaptIntroducer', text: 'adapt'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'base', has_error: yes},
// CHECK:STDOUT:       {kind: 'AdaptDecl', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
