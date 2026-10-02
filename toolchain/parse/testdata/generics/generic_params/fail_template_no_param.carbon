// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/fail_template_no_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/fail_template_no_param.carbon

// --- fail_template_before_type.carbon

// CHECK:STDERR: fail_template_before_type.carbon:[[@LINE+2]]:15-17: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR:
fn A(template i32);

// --- fail_template_no_name.carbon

// CHECK:STDERR: fail_template_no_name.carbon:[[@LINE+2]]:14: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR:
fn C(template);

// --- fail_reserved_word_after_generic.carbon

// CHECK:STDERR: fail_reserved_word_after_generic.carbon:[[@LINE+2]]:22: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR:
fn E(generic template: i32);

// CHECK:STDOUT: - filename: fail_template_before_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'A'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'i32' has_error
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'i32' has_error
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'i32' has_error
// CHECK:STDOUT:   │ │ ├─LetBindingPattern 'template' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_template_no_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'C'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature ')' has_error
// CHECK:STDOUT:   │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ ├─LetBindingPattern 'template' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_reserved_word_after_generic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'E'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'template'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
