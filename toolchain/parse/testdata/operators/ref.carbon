// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/ref.carbon

// --- ref_precedence.carbon

var x: i32 = ref y as i32;

// --- fail_if_ref.carbon

// CHECK:STDERR: fail_if_ref.carbon:[[@LINE+2]]:18-19: error: parentheses are required around this unary `if` operator [UnaryOperatorRequiresParentheses]
// CHECK:STDERR:
var x: i32 = ref if b then c else d;
var x: i32 = ref (if b then c else d);
var x: i32 = if b then ref c else ref d;

// CHECK:STDOUT: - filename: ref_precedence.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_if_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │   ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ ╭─IfExprElse 'else'
// CHECK:STDOUT:   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │   │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   ├─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ╭─ParenExpr ')'
// CHECK:STDOUT:   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │ ╭─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ ├─IfExprElse 'else'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
