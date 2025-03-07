// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/var_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/var_pattern.carbon

// --- basic.carbon

let var x: () = ();

// --- tuple.carbon

let (x: (), var y: ()) = ((), ());

// --- function.carbon

fn F(x: (), var y: ());

// --- fail_nested.carbon

// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:25: error: `var` nested within another `var` [NestedVar]
// CHECK:STDERR: let (x: (), var (y: (), var z: ())) = ((), ((), ()));
// CHECK:STDERR:                         ^~~
// CHECK:STDERR:
let (x: (), var (y: (), var z: ())) = ((), ((), ()));

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: tuple.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: function.carbon
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
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'z'},
// CHECK:STDOUT:                   {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 35},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
