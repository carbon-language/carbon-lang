// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/call.carbon

// --- basic.carbon

fn F() {
  a.b.f(c.d, (e)).g();
}

// --- ref_args.carbon

fn F() {
  G(ref x);
  G(ref x, y);
  G(x, ref y);
}

// --- nested_ref.carbon

fn F() {
  G(x, (ref x,));
  G(x, (ref y, z));
  G(x, (y, ref z));

  G(x, {.y = ref y, .z = z});
}

// --- fail_ref_out_of_place.carbon

fn F() {
  (ref x,);
  G(ref x);
  // CHECK:STDERR: fail_ref_out_of_place.carbon:[[@LINE+4]]:4: error: found `ref` in unexpected position [UnexpectedRef]
  // CHECK:STDERR:   (ref x);
  // CHECK:STDERR:    ^~~
  // CHECK:STDERR:
  (ref x);
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                       {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:                       {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:                     {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:                     {kind: 'IdentifierNameNotBeforeParams', text: 'f'},
// CHECK:STDOUT:                   {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'CallExprStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeParams', text: 'd'},
// CHECK:STDOUT:                 {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:                   {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'e'},
// CHECK:STDOUT:                 {kind: 'ParenExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CallExpr', text: ')', subtree_size: 14},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'g'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 16},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: ref_args.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: nested_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:             {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 11},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:             {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'z'},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:             {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'z'},
// CHECK:STDOUT:             {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:               {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'z'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'z'},
// CHECK:STDOUT:             {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 17},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 59},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_ref_out_of_place.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefTag', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
