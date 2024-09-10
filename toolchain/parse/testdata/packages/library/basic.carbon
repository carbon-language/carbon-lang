// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/library/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/library/basic.carbon

// --- default.carbon

library default;

// --- named.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDOUT: - filename: default.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'DefaultLibrary', text: 'default'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: named.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"named"'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
