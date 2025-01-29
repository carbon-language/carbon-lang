// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/tuple/access/value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/tuple/access/value_access.carbon

let x: i32 = tuple.0;

// CHECK:STDOUT: - filename: value_access.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'tuple'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
