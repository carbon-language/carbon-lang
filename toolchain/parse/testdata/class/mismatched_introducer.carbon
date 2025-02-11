// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/mismatched_introducer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/mismatched_introducer.carbon

// Note: invalid modifiers are diagnosed in check not parse.
abstract interface I;
base fn F();
abstract base class C;

// CHECK:STDOUT: - filename: mismatched_introducer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
