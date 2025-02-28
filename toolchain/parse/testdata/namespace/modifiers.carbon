// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/modifiers.carbon

private extern abstract base default final namespace Foo;

impl namespace Foo;

// CHECK:STDOUT: - filename: modifiers.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'PrivateModifier', text: 'private'},
// CHECK:STDOUT:       {kind: 'ExternModifier', text: 'extern'},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:       {kind: 'DefaultModifier', text: 'default'},
// CHECK:STDOUT:       {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
