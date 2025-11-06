// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_generic_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_generic_ref.carbon

// CHECK:STDERR: fail_generic_ref.carbon:[[@LINE+4]]:5: error: expected `:` binding after `ref` [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: let ref X:! i32 = 0;
// CHECK:STDERR:     ^~~
// CHECK:STDERR:
let ref X:! i32 = 0;

// CHECK:STDOUT: - filename: fail_generic_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'X'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPatternStart', text: ':!', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
