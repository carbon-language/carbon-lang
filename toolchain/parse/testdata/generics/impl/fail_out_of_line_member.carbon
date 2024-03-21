// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Decide if we want to support this syntax. See #3763.

interface Interface {
  fn F();
}

impl bool as Interface {
  fn F();
}

// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+4]]:4: ERROR: `fn` introducer should be followed by a name.
// CHECK:STDERR: fn (bool as Interface).F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn (bool as Interface).F() {}

class C {
  impl Self as Interface {
    fn F();
  }
}

// TODO: The error recovery here is not very good. The `(` is treated as
// starting the function parameter list.
// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+11]]:6: ERROR: Expected identifier after `.`.
// CHECK:STDERR: fn C.(Self as Interface).F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+7]]:7: ERROR: Expected binding pattern.
// CHECK:STDERR: fn C.(Self as Interface).F() {}
// CHECK:STDERR:       ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+3]]:25: ERROR: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: fn C.(Self as Interface).F() {}
// CHECK:STDERR:                         ^
fn C.(Self as Interface).F() {}

// CHECK:STDOUT: - filename: fail_out_of_line_member.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: '(', has_error: yes},
// CHECK:STDOUT:       {kind: 'QualifiedName', text: '.', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'Self', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: 'Self', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: '}', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
