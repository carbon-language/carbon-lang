// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/observe.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/observe.carbon

// --- base.carbon

observe A == B == C;

// --- impls.carbon

observe T impls I;

// --- equal_equal_impls.carbon

observe T == U impls I;
observe A == B == C == D impls I;

// --- statement.carbon

fn F() {
    observe A == B;
}

interface I {
    observe A == B;
}

// CHECK:STDOUT: - filename: base.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:         {kind: 'ObserveEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:       {kind: 'ObserveEqualEqual', text: '==', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ObserveImpls', text: 'impls', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: equal_equal_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'U'},
// CHECK:STDOUT:         {kind: 'ObserveEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ObserveImpls', text: 'impls', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:             {kind: 'ObserveEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:           {kind: 'ObserveEqualEqual', text: '==', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'D'},
// CHECK:STDOUT:         {kind: 'ObserveEqualEqual', text: '==', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ObserveImpls', text: 'impls', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: statement.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:         {kind: 'ObserveEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ObserveDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'I'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:         {kind: 'ObserveEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ObserveDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
