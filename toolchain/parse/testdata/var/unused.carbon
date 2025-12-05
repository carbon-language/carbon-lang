// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/unused.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/unused.carbon

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


// --- fail_unused.carbon

// CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:5: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR: var ref unused y: i32;
// CHECK:STDERR:     ^~~
// CHECK:STDERR:
var ref unused y: i32;


// --- fail_var_ref_unused.carbon

// CHECK:STDERR: fail_var_ref_unused.carbon:[[@LINE+4]]:5: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR: var ref unused y: i32;
// CHECK:STDERR:     ^~~
// CHECK:STDERR:
var ref unused y: i32;

// --- fail_var_ref_unused_function.carbon

// CHECK:STDERR: fail_var_ref_unused_function.carbon:[[@LINE+4]]:17: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR: fn F(x: (), var ref unused y: ());
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
fn F(x: (), var ref unused y: ());

// CHECK:STDERR: fail_var_ref_unused_function.carbon:[[@LINE+4]]:24: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR: fn G(x: (), var unused ref y: ());
// CHECK:STDERR:                        ^~~
// CHECK:STDERR:
fn G(x: (), var unused ref y: ());

// CHECK:STDERR: fail_var_ref_unused_function.carbon:[[@LINE+4]]:24: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR: fn I(x: (), unused var ref y: ());
// CHECK:STDERR:                        ^~~
// CHECK:STDERR:
fn I(x: (), unused var ref y: ());

// CHECK:STDOUT: - filename: let_var_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'w'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'c'},
// CHECK:STDOUT:               {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 21},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'z'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'w'},
// CHECK:STDOUT:             {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 't'},
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
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
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
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
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
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
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
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
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
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
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
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'unused', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'unused', has_error: yes},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_var_ref_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'unused', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'unused', has_error: yes},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_var_ref_unused_function.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'unused', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'unused', has_error: yes},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'UnusedPattern', text: 'unused', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'I'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
