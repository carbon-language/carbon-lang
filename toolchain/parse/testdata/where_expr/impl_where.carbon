// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon

interface Interface {
  let T:! type;
}

impl i32 as Interface where .T = () {
}

impl bool as Interface where .T = i32;

impl () as (Interface where .T = {}) {}

impl {} as (Interface where .T = ()) where .T = ();

impl f64 as Interface where .T = (type where .Self impls type) {
}

// CHECK:STDOUT: - filename: impl_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:       {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:         {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'TypeImplAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:               {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:                 {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                   {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'WhereOperand', text: 'where', subtree_size: 11},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'WhereExpr', text: 'where', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', subtree_size: 22},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'FloatTypeLiteral', text: 'f64'},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                   {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:                 {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                     {kind: 'SelfTypeName', text: 'Self'},
// CHECK:STDOUT:                   {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                   {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:                 {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:               {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
