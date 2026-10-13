// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_no_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_no_type.carbon

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:7: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var x = 5;
// CHECK:STDERR:       ^
// CHECK:STDERR:
var x = 5;

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:6: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var y;
// CHECK:STDERR:      ^
// CHECK:STDERR:
var y;

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:17: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var some_object = ClassType;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
var some_object = ClassType;

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:7: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var z i32 = 5;
// CHECK:STDERR:       ^~~
// CHECK:STDERR:
var z i32 = 5;

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:7: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var (a, b: i32) = (1, 2);
// CHECK:STDERR:       ^
// CHECK:STDERR:
var (a, b: i32) = (1, 2);

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:15: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var (c: i32, d) = (1, 2);
// CHECK:STDERR:               ^
// CHECK:STDERR:
var (c: i32, d) = (1, 2);

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:12: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var {.a = e} = {.a = 1};
// CHECK:STDERR:            ^
// CHECK:STDERR:
var {.a = e} = {.a = 1};

// CHECK:STDERR: fail_no_type.carbon:[[@LINE+12]]:11: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn F[var T](var u, var v = 1);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_type.carbon:[[@LINE+8]]:18: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn F[var T](var u, var v = 1);
// CHECK:STDERR:                  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:26: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn F[var T](var u, var v = 1);
// CHECK:STDERR:                          ^
// CHECK:STDERR:
fn F[var T](var u, var v = 1);

fn M(n: i32) {
  match (n) {
    // CHECK:STDERR: fail_no_type.carbon:[[@LINE+4]]:16: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
    // CHECK:STDERR:     case var w => {}
    // CHECK:STDERR:                ^~
    // CHECK:STDERR:
    case var w => {}
  }
}

// CHECK:STDOUT: - filename: fail_no_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern 'x' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '5'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern 'y' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'some_object'
// CHECK:STDOUT:   │ │   ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern 'some_object' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'ClassType'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │   ├─InvalidParse 'i32' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse 'i32' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern 'z' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '5'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   ├─VarBindingPattern 'a' has_error
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ╭─TuplePattern ')' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'd'
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─VarBindingPattern 'd' has_error
// CHECK:STDOUT:   │ │ ╭─TuplePattern ')' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'e'
// CHECK:STDOUT:   │ │   │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │   │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern 'e' has_error
// CHECK:STDOUT:   │ │   ├─StructPatternDesignatedField '=' has_error
// CHECK:STDOUT:   │ │ ╭─StructPattern '}'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │   ├─InvalidParse ']' has_error
// CHECK:STDOUT:   │ │ │   ├─InvalidParse ']' has_error
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern 'T' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─ImplicitParamList ']' has_error
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'u'
// CHECK:STDOUT:   │ │ │   ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │ │   ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern 'u' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameNotBeforeSignature 'v'
// CHECK:STDOUT:   │ │ │     ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │ │     ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │ │   ╭─VarBindingPattern 'v' has_error
// CHECK:STDOUT:   │ │ │ ╭─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'M'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─MatchIntroducer 'match'
// CHECK:STDOUT:   │ │   │ ╭─MatchConditionStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │   ├─MatchCondition ')'
// CHECK:STDOUT:   │ │ ╭─MatchStatementStart '{'
// CHECK:STDOUT:   │ │ │     ╭─MatchCaseIntroducer 'case'
// CHECK:STDOUT:   │ │ │     │   ╭─IdentifierNameNotBeforeSignature 'w'
// CHECK:STDOUT:   │ │ │     │   ├─InvalidParse '=>' has_error
// CHECK:STDOUT:   │ │ │     │   ├─InvalidParse '=>' has_error
// CHECK:STDOUT:   │ │ │     │ ╭─VarBindingPattern 'w' has_error
// CHECK:STDOUT:   │ │ │     ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ │ │   ╭─MatchCase '=>' has_error
// CHECK:STDOUT:   │ │ │ ╭─MatchHandlerStart '=>' has_error
// CHECK:STDOUT:   │ │ ├─MatchHandler '=>' has_error
// CHECK:STDOUT:   │ ├─MatchStatement '}'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
