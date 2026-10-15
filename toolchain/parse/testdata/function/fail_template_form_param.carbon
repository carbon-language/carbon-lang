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
// CHECK:STDERR: fail_template_form_param.carbon:[[@LINE+2]]:6-13: error: `template` is only allowed on a generic binding [ExpectedGenericBindingPatternAfterTemplate]
// CHECK:STDERR:
fn F(template x:? Form);

// CHECK:STDOUT: - filename: fail_template_form_param.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':?' has_error
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'Form'
// CHECK:STDOUT:   │ │ ├─FormBindingPattern ':?' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
