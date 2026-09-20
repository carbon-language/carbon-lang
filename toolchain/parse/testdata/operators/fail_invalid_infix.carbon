// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_invalid_infix.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_invalid_infix.carbon

// CHECK:STDERR: fail_invalid_infix.carbon:[[@LINE+2]]:19: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
var a: i32 = n == ;
// CHECK:STDERR: fail_invalid_infix.carbon:[[@LINE+2]]:14-15: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
var b: i32 = == n;
// CHECK:STDERR: fail_invalid_infix.carbon:[[@LINE+4]]:14-15: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_infix.carbon:[[@LINE+2]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
var c: i32 = == ;

// CHECK:STDOUT: - filename: fail_invalid_infix.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ ├─InfixOperatorEqualEqual '==' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─InvalidParse '==' has_error
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ ├─InfixOperatorEqualEqual '==' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─InvalidParse '==' has_error
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ ├─InfixOperatorEqualEqual '==' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
