// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/fail_missing_expr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/fail_missing_expr.carbon

// --- fail_missing_expr_after_observe.carbon

// CHECK:STDERR: fail_missing_expr_after_observe.carbon:[[@LINE+4]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: observe impls I;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR:
observe impls I;

// --- fail_missing_expr_after_impls.carbon

// CHECK:STDERR: fail_missing_expr_after_impls.carbon:[[@LINE+4]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR: observe T impls ;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
observe T impls ;

// --- fail_missing_impls.carbon

// CHECK:STDERR: fail_missing_impls.carbon:[[@LINE+4]]:11: error: observe should use `==` or `impls` operator [ExpectedObserveOperator]
// CHECK:STDERR: observe T ;
// CHECK:STDERR:           ^
// CHECK:STDERR:
observe T ;

// CHECK:STDOUT: - filename: fail_missing_expr_after_observe.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'impls', has_error: yes},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ObserveImpls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing_expr_after_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'ObserveImpls', text: 'impls', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
