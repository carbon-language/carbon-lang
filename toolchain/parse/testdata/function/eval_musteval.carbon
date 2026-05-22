// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/eval_musteval.carbon

// --- eval.carbon

eval fn F();

// --- musteval.carbon

musteval fn F();

// --- mixed.carbon

eval musteval fn F();

// CHECK:STDOUT: - filename: eval.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'EvalModifier', text: 'eval'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: musteval.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'MustEvalModifier', text: 'musteval'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: mixed.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'EvalModifier', text: 'eval'},
// CHECK:STDOUT:       {kind: 'MustEvalModifier', text: 'musteval'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
