// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/params/fail_unused_after_phase_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/params/fail_unused_after_phase_keyword.carbon

// --- fail_generic_before_unused.carbon

// CHECK:STDERR: fail_generic_before_unused.carbon:[[@LINE+2]]:14-19: error: `unused` must be written before `generic` [UnusedAfterBindingModifier]
// CHECK:STDERR:
fn F(generic unused T: type);

// --- fail_runtime_before_unused.carbon

// CHECK:STDERR: fail_runtime_before_unused.carbon:[[@LINE+2]]:14-19: error: `unused` must be written before `runtime` [UnusedAfterBindingModifier]
// CHECK:STDERR:
fn G[runtime unused T: type]();

// --- fail_redundant_runtime_before_unused.carbon

// CHECK:STDERR: fail_redundant_runtime_before_unused.carbon:[[@LINE+2]]:6-12: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR:
fn H(runtime unused x: i32);

// CHECK:STDOUT: - filename: fail_generic_before_unused.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ │   ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ├─UnusedPattern 'unused' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_runtime_before_unused.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │   ╭─RuntimeBindingName 'runtime'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ │   ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ ╭─LetBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ├─UnusedPattern 'unused' has_error
// CHECK:STDOUT:   │ ├─ImplicitParamList ']' has_error
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_redundant_runtime_before_unused.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ╭─RuntimeBindingName 'runtime'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─UnusedPattern 'unused'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
