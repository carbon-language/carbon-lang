// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/require.carbon

// --- implicit_self_impls.carbon

interface Y {}

interface Z {
  require impls Y;
}

// --- explicit_self_impls.carbon

interface Y {}

interface Z {
  require Self impls Y;
}

// --- impls_where.carbon

interface Y { let Y1:! type; }

interface Z {
  require impls Y where .Y1 = ();
}

// --- other_impls.carbon

interface Y {}

interface Z(T:! type) {
  require T impls Y;
}

// --- extend.carbon

interface Y {}

interface Z {
  extend require impls Y;
}

// --- type_and.carbon

interface Y {}

interface Z {
  require impls Y & Y;
}

// --- fail_missing_impls.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_missing_impls.carbon:[[@LINE+4]]:12: error: expected `impls` in `require` declaration [RequireExpectedImpls]
  // CHECK:STDERR:   require Y;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  require Y;
}

// --- fail_wrong_keyword.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_wrong_keyword.carbon:[[@LINE+4]]:11: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   require as Y;
  // CHECK:STDERR:           ^~
  // CHECK:STDERR:
  require as Y;
}

// --- fail_as_in_self.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_as_in_self.carbon:[[@LINE+4]]:16: error: expected `impls` in `require` declaration [RequireExpectedImpls]
  // CHECK:STDERR:   require Self as Y impls Y;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  require Self as Y impls Y;
}

// CHECK:STDOUT: - filename: implicit_self_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'RequireDefaultSelfImpls', text: 'impls'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: explicit_self_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'RequireTypeImpls', text: 'impls', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: impls_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Y1'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantNameAndType', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'RequireDefaultSelfImpls', text: 'impls'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'Y1'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: other_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Z'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPatternStart', text: ':!', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'RequireTypeImpls', text: 'impls', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: extend.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'RequireDefaultSelfImpls', text: 'impls'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: type_and.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'RequireDefaultSelfImpls', text: 'impls'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAmp', text: '&', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_wrong_keyword.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'as', has_error: yes},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_as_in_self.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'RequireIntroducer', text: 'require'},
// CHECK:STDOUT:         {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:       {kind: 'RequireDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
