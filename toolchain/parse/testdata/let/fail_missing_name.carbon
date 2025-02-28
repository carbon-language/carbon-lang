// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_missing_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_missing_name.carbon

// --- fail_runtime_binding.carbon

// CHECK:STDERR: fail_runtime_binding.carbon:[[@LINE+4]]:5: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: let : i32 = 4;
// CHECK:STDERR:     ^
// CHECK:STDERR:
let : i32 = 4;

// --- fail_complietime_binding.carbon

// CHECK:STDERR: fail_complietime_binding.carbon:[[@LINE+4]]:5: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: let :! bool = true;
// CHECK:STDERR:     ^~
// CHECK:STDERR:
let :! bool = true;

// CHECK:STDOUT: - filename: fail_runtime_binding.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: ':', has_error: yes},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_complietime_binding.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: ':!', has_error: yes},
// CHECK:STDOUT:         {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:       {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
