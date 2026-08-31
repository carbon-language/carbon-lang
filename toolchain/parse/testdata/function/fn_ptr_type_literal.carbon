// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/fn_ptr_type_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/fn_ptr_type_literal.carbon

// --- ptr_type_literal.carbon

fn F() {
  __fn_ptr() ->? ();
  __fn_ptr(a, b, c) ->? d;
}

// --- fail_no_return_form.carbon

fn F() {
  // CHECK:STDERR: fail_no_return_form.carbon:[[@LINE+4]]:13: error: Missing `->?` return form for function pointer type literal [ExpectedReturnFormInFunctionPointerType]
  // CHECK:STDERR:   __fn_ptr();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  __fn_ptr();
}


// --- fail_missing_question_mark.carbon

fn F() {
  // CHECK:STDERR: fail_missing_question_mark.carbon:[[@LINE+4]]:21: error: Missing `->?` return form for function pointer type literal [ExpectedReturnFormInFunctionPointerType]
  // CHECK:STDERR:   __fn_ptr(a, b, c) -> d;
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR:
  __fn_ptr(a, b, c) -> d;
}

// CHECK:STDOUT: - filename: ptr_type_literal.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   ╭─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │   ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteral ')'
// CHECK:STDOUT:   │ │   ├─FnPtrTypeLiteralReturn '->?'
// CHECK:STDOUT:   │ │ ╭─FnPtrTypeLiteral '__fn_ptr'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │     ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   ╭─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   ├─FnPtrTypeLiteralReturn '->?'
// CHECK:STDOUT:   │ │ ╭─FnPtrTypeLiteral '__fn_ptr'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_return_form.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   ╭─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─FnPtrTypeLiteral '__fn_ptr' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_question_mark.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │       ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │       ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │       ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │     ╭─TupleLiteral ')'
// CHECK:STDOUT:   │ │   ╭─FnPtrTypeLiteral '__fn_ptr' has_error
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'd'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
