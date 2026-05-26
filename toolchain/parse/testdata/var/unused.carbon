// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/unused.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/unused.carbon

// --- fail_unused_does_not_start_declaration.carbon

// CHECK:STDERR: fail_unused_does_not_start_declaration.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: unused y: i32;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
unused y: i32;

// --- let_var_unused.carbon
var unused y: i32;
let unused y: i32;
let unused ref w: i32 = 0;

fn F(unused x: i32, var unused b: i32, unused ref c: i32) {
  var unused y: i32;
  let unused z: i32;
  let unused ref w: i32 = 0;
  let unused ref t: str = "hello";
}

// --- unused_tuple.carbon
let (x: (), var unused y: ()) = ((), ());
let (x: (), unused var y: ()) = ((), ());

let unused (x: (), y: ()) = ((), ());
let var (x: (), unused y: ()) = ((), ());
let var unused (x: (), y: ()) = ((), ());

// --- var_unused_function.carbon
fn F(x: (), var unused y: ());

// --- var_unused_match.carbon
fn F() -> i32 {
  var x: i32 = 3;
  match (f(x)) {
    case unused var (a: i32, b: i32) => { return 0; }
    case (a: i32) if (a < 0) => { return 2; }
    case var unused a: i32 if (a != x) => { return 3; }
    default => { return 4; }
  }
  return 0;
}

// --- unused_var_function.carbon
fn F(x: (), unused var y: ());

// --- fail_ref_unused.carbon

// CHECK:STDERR: fail_ref_unused.carbon:[[@LINE+4]]:10: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn F(ref unused y: i32);
// CHECK:STDERR:          ^~~~~~
// CHECK:STDERR:
fn F(ref unused y: i32);

// --- fail_nested.carbon

// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:12: error: `unused` nested within another `unused` [NestedUnused]
// CHECK:STDERR: var unused unused x: i32 = 0;
// CHECK:STDERR:            ^~~~~~
// CHECK:STDERR:
var unused unused x: i32 = 0;

// --- fail_nested_parens.carbon

// CHECK:STDERR: fail_nested_parens.carbon:[[@LINE+4]]:13: error: `unused` nested within another `unused` [NestedUnused]
// CHECK:STDERR: var unused (unused y: i32) = (0,);
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR:
var unused (unused y: i32) = (0,);

// CHECK:STDOUT: - filename: fail_unused_does_not_start_declaration.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'unused', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: let_var_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'w'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'c'},
// CHECK:STDOUT:               {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 21},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'z'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'w'},
// CHECK:STDOUT:             {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 't'},
// CHECK:STDOUT:             {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 53},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: unused_tuple.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'UnusedPattern', text: 'unused', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 22},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: var_unused_function.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: var_unused_match.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'f'},
// CHECK:STDOUT:               {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:             {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:               {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:                     {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:                       {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:                       {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                     {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:                     {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                       {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:                       {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                     {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'TuplePattern', text: ')', subtree_size: 9},
// CHECK:STDOUT:                 {kind: 'VariablePattern', text: 'var', subtree_size: 10},
// CHECK:STDOUT:               {kind: 'UnusedPattern', text: 'unused', subtree_size: 11},
// CHECK:STDOUT:             {kind: 'MatchCase', text: '=>', subtree_size: 13},
// CHECK:STDOUT:           {kind: 'MatchHandlerStart', text: '{', subtree_size: 14},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchHandler', text: '}', subtree_size: 18},
// CHECK:STDOUT:               {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:                 {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:                   {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                 {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'ParenPattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:                   {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:                 {kind: 'InfixOperatorLess', text: '<', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuard', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCase', text: '=>', subtree_size: 13},
// CHECK:STDOUT:           {kind: 'MatchHandlerStart', text: '{', subtree_size: 14},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchHandler', text: '}', subtree_size: 18},
// CHECK:STDOUT:               {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:                     {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:                     {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                   {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:               {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:                 {kind: 'InfixOperatorExclaimEqual', text: '!=', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuard', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCase', text: '=>', subtree_size: 13},
// CHECK:STDOUT:           {kind: 'MatchHandlerStart', text: '{', subtree_size: 14},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchHandler', text: '}', subtree_size: 18},
// CHECK:STDOUT:               {kind: 'MatchDefaultIntroducer', text: 'default'},
// CHECK:STDOUT:             {kind: 'MatchDefault', text: '=>', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'MatchHandlerStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchHandler', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 70},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 89},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: unused_var_function.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_ref_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'unused', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'unused', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_nested_parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ParenPattern', text: ')', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
