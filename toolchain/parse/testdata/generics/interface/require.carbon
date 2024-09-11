// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/require.carbon

// --- require_name.carbon

interface Foo {
  require i32 impls Bar;
}

// --- require_self.carbon

interface Foo {
  require Self impls Bar;
}

// --- fail_empty_require.carbon

interface Foo {
  // CHECK:STDERR: fail_empty_require.carbon:[[@LINE+4]]:10: ERROR: Expected expression.
  // CHECK:STDERR:   require;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  require;
}

// --- fail_single_arg_require.carbon

interface Foo {
  // CHECK:STDERR: fail_single_arg_require.carbon:[[@LINE+4]]:14: ERROR: Expected `impls` in `require` declaration.
  // CHECK:STDERR:   require i32;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  require i32;
}

// --- fail_single_arg2_require.carbon

interface Foo {
  // CHECK:STDERR: fail_single_arg2_require.carbon:[[@LINE+8]]:11: ERROR: Expected expression.
  // CHECK:STDERR:   require impls;
  // CHECK:STDERR:           ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_single_arg2_require.carbon:[[@LINE+4]]:16: ERROR: Expected expression.
  // CHECK:STDERR:   require impls;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  require impls;
}

// --- fail_double_arg_require.carbon

interface Foo {
  // CHECK:STDERR: fail_double_arg_require.carbon:[[@LINE+4]]:20: ERROR: Expected expression.
  // CHECK:STDERR:   require i32 impls;
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  require i32 impls;
}

// --- fail_double_arg2_require.carbon

interface Foo {
  // CHECK:STDERR: fail_double_arg2_require.carbon:[[@LINE+4]]:11: ERROR: Expected expression.
  // CHECK:STDERR:   require impls i32;
  // CHECK:STDERR:           ^~~~~
  // CHECK:STDERR:
  require impls i32;
}

// --- fail_extra_arg_require.carbon

interface Foo {
  // CHECK:STDERR: fail_extra_arg_require.carbon:[[@LINE+4]]:15: ERROR: Expected `impls` in `require` declaration.
  // CHECK:STDERR:   require i32 i32 impls Bar;
  // CHECK:STDERR:               ^~~
  // CHECK:STDERR:
  require i32 i32 impls Bar;
}

// --- fail_extra_arg2_require.carbon

interface Foo {
  // CHECK:STDERR: fail_extra_arg2_require.carbon:[[@LINE+4]]:25: ERROR: `require` declarations must end with a `;`.
  // CHECK:STDERR:   require i32 impls Bar Bar;
  // CHECK:STDERR:                         ^~~
  // CHECK:STDERR:
  require i32 impls Bar Bar;
}

// --- fail_extra_arg3_require.carbon

interface Foo {
  // CHECK:STDERR: fail_extra_arg3_require.carbon:[[@LINE+8]]:21: ERROR: Expected expression.
  // CHECK:STDERR:   require i32 impls impls Bar;
  // CHECK:STDERR:                     ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_extra_arg3_require.carbon:[[@LINE+4]]:21: ERROR: `require` declarations must end with a `;`.
  // CHECK:STDERR:   require i32 impls impls Bar;
  // CHECK:STDERR:                     ^~~~~
  // CHECK:STDERR:
  require i32 impls impls Bar;
}

// --- fail_wrong_order_require.carbon

interface Foo {
  // CHECK:STDERR: fail_wrong_order_require.carbon:[[@LINE+8]]:11: ERROR: Expected expression.
  // CHECK:STDERR:   require impls i32 Bar;
  // CHECK:STDERR:           ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_wrong_order_require.carbon:[[@LINE+4]]:21: ERROR: `require` declarations must end with a `;`.
  // CHECK:STDERR:   require impls i32 Bar;
  // CHECK:STDERR:                     ^~~
  // CHECK:STDERR:
  require impls i32 Bar;
}

// --- fail_wrong_order2_require.carbon

interface Foo {
  // CHECK:STDERR: fail_wrong_order2_require.carbon:[[@LINE+3]]:15: ERROR: Expected `impls` in `require` declaration.
  // CHECK:STDERR:   require i32 Bar impls;
  // CHECK:STDERR:               ^~~
  require i32 Bar impls;
}

// CHECK:STDOUT: - filename: require_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: require_self.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_empty_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_single_arg_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_single_arg2_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'impls', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_double_arg_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_double_arg2_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'impls', has_error: yes},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_extra_arg_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_extra_arg2_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_extra_arg3_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'impls', has_error: yes},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_wrong_order_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'impls', has_error: yes},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'Impls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_wrong_order2_require.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'Require', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
