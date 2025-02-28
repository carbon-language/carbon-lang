// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/template_addr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/template_addr.carbon

// --- fail_wrong_order.carbon

// CHECK:STDERR: fail_wrong_order.carbon:[[@LINE+4]]:17: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn foo(template addr a:! i32);
// CHECK:STDERR:                 ^~~~
// CHECK:STDERR:
fn foo(template addr a:! i32);

// --- correct_order.carbon

fn bar(addr template b:! i32);

// CHECK:STDOUT: - filename: fail_wrong_order.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'addr', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'addr', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'template', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: correct_order.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'bar'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:             {kind: 'TemplateBindingName', text: 'template', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'Addr', text: 'addr', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
