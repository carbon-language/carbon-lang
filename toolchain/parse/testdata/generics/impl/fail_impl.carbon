// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:10: ERROR: Expected `as` in `impl` declaration.
// CHECK:STDERR: impl foo bar;
// CHECK:STDERR:          ^~~
impl foo bar;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:12: ERROR: Expected expression.
// CHECK:STDERR: impl foo as;
// CHECK:STDERR:            ^
impl foo as;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:17: ERROR: `impl` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: impl foo as bar baz;
// CHECK:STDERR:                 ^~~
impl foo as bar baz;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:13: ERROR: Expected `[` after `forall` in `impl` declaration.
// CHECK:STDERR: impl forall foo;
// CHECK:STDERR:             ^~~
impl forall foo;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:19: ERROR: Expected `as` in `impl` declaration.
// CHECK:STDERR: impl forall [] foo;
// CHECK:STDERR:                   ^
impl forall [] foo;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:27: ERROR: Expected `as` in `impl` declaration.
// CHECK:STDERR: impl forall [T:! type] foo;
// CHECK:STDERR:                           ^
impl forall [T:! type] foo;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:28: ERROR: Expected `as` in `impl` declaration.
// CHECK:STDERR: impl forall [T:! type] foo bar;
// CHECK:STDERR:                            ^~~
impl forall [T:! type] foo bar;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:35: ERROR: `impl` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: impl forall [T:! type] foo as bar baz;
// CHECK:STDERR:                                   ^~~
impl forall [T:! type] foo as bar baz;

impl

// CHECK:STDERR: fail_impl.carbon:[[@LINE+60]]:21: ERROR: Expected expression.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'bar'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ImplForall', text: 'forall', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplForall', text: 'forall', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplForall', text: 'forall', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplForall', text: 'forall', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'bar'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: 'impl', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
