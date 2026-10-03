// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/terse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/terse.carbon

// --- basic.carbon

fn F(x: i32) => x;

// --- fail_return_type.carbon

// CHECK:STDERR: fail_return_type.carbon:[[@LINE+4]]:21: error: cannot specify a return type with `=>` [ReturnTypeInTerseFunction]
// CHECK:STDERR: fn G(x: i32) -> i32 => x;
// CHECK:STDERR:                     ^~
// CHECK:STDERR:
fn G(x: i32) -> i32 => x;

// CHECK:STDERR: fail_return_type.carbon:[[@LINE+4]]:30: error: cannot specify a return type with `=>` [ReturnTypeInTerseFunction]
// CHECK:STDERR: fn H(ref x: i32) ->? ref i32 => x;
// CHECK:STDERR:                              ^~
// CHECK:STDERR:
fn H(ref x: i32) ->? ref i32 => x;

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '=>'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   ├─FunctionTerseDefinition ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_return_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '=>' has_error
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   ├─FunctionTerseDefinition ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │   │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │   ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ╭─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │   ├─ReturnForm '->?'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '=>' has_error
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   ├─FunctionTerseDefinition ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
