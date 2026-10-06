// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/value_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/value_literals.carbon

// --- int_literals.carbon

// 8 and 9 trigger special behavior in APInt when mishandling signed versus
// unsigned, so we pay extra attention to those.
var ints: array(i32, 5) = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
);

// --- real_literals.carbon

var reals: array(f64, 7) = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
    39999999999999999993.0e39999999999999999993,
);

// --- bool_literal.carbon

var f: bool = false;
var t: bool = true;

// --- char_literal.carbon

var c: Core.Char = 'c';

// --- string_literal.carbon

var test_str: str = "Test";

// CHECK:STDOUT: - filename: int_literals.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'ints'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '5'
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─IntLiteral '8'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '9'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '0x8'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '0b1000'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '39999999999999999993'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: real_literals.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'reals'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─FloatTypeLiteral 'f64'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '7'
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─RealLiteral '0.9'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '8.0'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '80.0'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '1.0e7'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '1.0e8'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '1.0e-8'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─RealLiteral '39999999999999999993.0e39999999999999999993'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: bool_literal.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'f'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─BoolLiteralFalse 'false'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 't'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: char_literal.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─CoreNameExpr 'Core'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameNotBeforeSignature 'Char'
// CHECK:STDOUT:   │ │   ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─CharLiteral ''c''
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: string_literal.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'test_str'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─StringLiteral '"Test"'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
