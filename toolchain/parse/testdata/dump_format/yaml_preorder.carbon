// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --parse-dump-format=yaml-preorder
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/dump_format/yaml_preorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/dump_format/yaml_preorder.carbon

// --- fail_yaml_preorder.carbon

// CHECK:STDERR: fail_yaml_preorder.carbon:[[@LINE+2]]:9: error: expected `.field: field_type` or `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR:
var x: {,} = {};

// CHECK:STDOUT: - filename: fail_yaml_preorder.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FileStart', text: ''},
// CHECK:STDOUT:     {node_index: 13, kind: 'VariableDecl', text: ';', subtree_size: 13, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {node_index: 9, kind: 'VariablePattern', text: 'var', subtree_size: 8, children: [
// CHECK:STDOUT:         {node_index: 8, kind: 'VarBindingPattern', text: ':', subtree_size: 7, children: [
// CHECK:STDOUT:           {node_index: 2, kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {node_index: 3, kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:           {node_index: 7, kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:             {node_index: 4, kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {node_index: 5, kind: 'InvalidParse', text: ',', has_error: yes},
// CHECK:STDOUT:             {node_index: 6, kind: 'InvalidParse', text: ',', has_error: yes}]}]}]},
// CHECK:STDOUT:       {node_index: 10, kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {node_index: 12, kind: 'StructLiteral', text: '}', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 11, kind: 'StructLiteralStart', text: '{'}]}]},
// CHECK:STDOUT:     {node_index: 14, kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
