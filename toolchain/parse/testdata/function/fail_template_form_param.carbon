// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/fail_template_form_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/fail_template_form_param.carbon

// `template` cannot be combined with a `:?` form binding.
// CHECK:STDERR: fail_template_form_param.carbon:[[@LINE+4]]:6: error: `template` is only allowed on a generic binding [ExpectedGenericBindingPatternAfterTemplate]
// CHECK:STDERR: fn F(template x:? Form);
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR:
fn F(template x:? Form);

// CHECK:STDOUT: - filename: fail_template_form_param.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Form'},
// CHECK:STDOUT:         {kind: 'FormBindingPattern', text: ':?', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
