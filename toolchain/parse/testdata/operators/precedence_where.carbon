// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_where.carbon

fn PostfixStar(T:! I* where .A* == .B*);

fn Ampersand(U:! J & K where C & D impls E & F);

fn BinaryPlus(V:! L + M where G & H impls N & O);

fn UnaryMinus(W:! -P where -Q impls -R);

// CHECK:STDOUT: - filename: precedence_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'PostfixStar'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:               {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:                 {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'PostfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeParams', text: 'B'},
// CHECK:STDOUT:                 {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'PostfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'RequirementEqualEqual', text: '==', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Ampersand'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'U'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'J'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:               {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 4},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'D'},
// CHECK:STDOUT:               {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'E'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'F'},
// CHECK:STDOUT:               {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'BinaryPlus'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'V'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'L'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'M'},
// CHECK:STDOUT:               {kind: 'InfixOperatorPlus', text: '+', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 4},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'H'},
// CHECK:STDOUT:               {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'N'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'O'},
// CHECK:STDOUT:               {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'UnaryMinus'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'W'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'P'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorMinus', text: '-', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Q'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorMinus', text: '-', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'R'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorMinus', text: '-', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
