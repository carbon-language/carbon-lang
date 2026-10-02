// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/params/fail_redundant_phase_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/params/fail_redundant_phase_keyword.carbon

// --- fail_redundant_generic_deduced.carbon

// CHECK:STDERR: fail_redundant_generic_deduced.carbon:[[@LINE+2]]:6-12: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR:
fn F[generic T: type]();

// --- fail_redundant_generic_class_param.carbon

// CHECK:STDERR: fail_redundant_generic_class_param.carbon:[[@LINE+2]]:9-15: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR:
class C(generic T: type) {}

// --- fail_redundant_runtime_explicit.carbon

// CHECK:STDERR: fail_redundant_runtime_explicit.carbon:[[@LINE+2]]:6-12: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR:
fn G(runtime x: i32);

// --- fail_redundant_generic_qualified.carbon

// CHECK:STDERR: fail_redundant_generic_qualified.carbon:[[@LINE+2]]:8-14: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR:
fn C.F[generic T: type]();

// --- fail_redundant_runtime_qualified.carbon

// CHECK:STDERR: fail_redundant_runtime_qualified.carbon:[[@LINE+2]]:8-14: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR:
fn C.G(runtime x: i32);

// CHECK:STDOUT: - filename: fail_redundant_generic_deduced.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_redundant_generic_class_param.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'C'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_redundant_runtime_explicit.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─RuntimeBindingName 'runtime'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_redundant_generic_qualified.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_redundant_runtime_qualified.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─RuntimeBindingName 'runtime'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
