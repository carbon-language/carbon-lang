// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/lambda/lambda.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/lambda/lambda.carbon

// --- basic.carbon

var a: auto = fn [T: type] (x: T) -> T { return x; };
var b: auto = fn (x: i32) => x;
var c: auto = fn { };

// --- fail_unconsumed_lambda_introducer.carbon

// Tests that the compiler does not crash when a LambdaIntroducer (`fn`)
// is left unconsumed inside an invalid binary operator sequence.

// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+12]]:10: error: whitespace missing around binary operator [BinaryOperatorRequiresWhitespace]
// CHECK:STDERR: require j%-fn->>
// CHECK:STDERR:          ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+8]]:16: error: expected expression [ExpectedExpr]
// CHECK:STDERR: require j%-fn->>
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+4]]:16: error: whitespace missing before binary operator [BinaryOperatorRequiresWhitespace]
// CHECK:STDERR: require j%-fn->>
// CHECK:STDERR:                ^
// CHECK:STDERR:
require j%-fn->>
// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+12]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+8]]:1: error: expected `=>` or `{` after return type [ExpectedLambdaBodyAfterReturnType]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unconsumed_lambda_introducer.carbon:[[@LINE+4]]:1: error: expected `impls` in `require` declaration [RequireExpectedImpls]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_expected_body.carbon

fn F() {
  // CHECK:STDERR: fail_expected_body.carbon:[[@LINE+4]]:20: error: expected `->`, `=>`, or `{` [ExpectedLambdaBody]
  // CHECK:STDERR:   var x: auto = fn ;
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  var x: auto = fn ;
}

// --- fail_expected_body_after_return_type.carbon

fn G() {
  // CHECK:STDERR: fail_expected_body_after_return_type.carbon:[[@LINE+4]]:27: error: expected `=>` or `{` after return type [ExpectedLambdaBodyAfterReturnType]
  // CHECK:STDERR:   var x: auto = fn -> i32 ;
  // CHECK:STDERR:                           ^
  // CHECK:STDERR:
  var x: auto = fn -> i32 ;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─Lambda 'fn'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ├─TerseBodyArrow '=>'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ ├─Lambda 'fn'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─Lambda 'fn'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_unconsumed_lambda_introducer.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'j'
// CHECK:STDOUT:   │ │ │   ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │   │   ╭─InvalidParse '>' has_error
// CHECK:STDOUT:   │ │ │   │   ├─InvalidParse '' has_error
// CHECK:STDOUT:   │ │ │   │ ╭─InfixOperatorGreater '>' has_error
// CHECK:STDOUT:   │ │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ │   ├─InvalidParse '' has_error
// CHECK:STDOUT:   │ │ │ ╭─Lambda 'fn' has_error
// CHECK:STDOUT:   │ │ ├─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ ├─InfixOperatorPercent '%'
// CHECK:STDOUT:   ├─RequireDecl '>' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_expected_body.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ├─Lambda 'fn' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_expected_body_after_return_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─LambdaIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ├─Lambda 'fn' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
