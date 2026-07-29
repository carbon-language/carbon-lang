// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/template.carbon

// --- basic.carbon

fn foo(template a: i32);

// --- fail_template_ref.carbon

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:17: error: `ref` is only allowed on a runtime binding [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn foo(template ref a: i32);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
fn foo(template ref a: i32);

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:17: error: `ref` is only allowed on a runtime binding [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn foo(template ref a: i32);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
fn foo(template ref a: i32);

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:12: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn foo(ref template a: i32);
// CHECK:STDERR:            ^~~~~~~~
// CHECK:STDERR:
fn foo(ref template a: i32);

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─TemplateBindingName 'template'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_template_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─TemplateBindingName 'template'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':' ❌
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':' ❌
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─TemplateBindingName 'template'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':' ❌
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':' ❌
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'template' ❌
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'template' ❌
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'template' ❌
// CHECK:STDOUT:   │ │ ├─LetBindingPattern 'ref' ❌
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
