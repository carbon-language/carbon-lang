// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+4]]:5: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var package.val: i32;
// CHECK:STDERR:     ^~~~~~~
// CHECK:STDERR:
var package.val: i32;

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace package.NS;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
namespace package.NS;

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+4]]:7: error: `class` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: class package.C {
// CHECK:STDERR:       ^~~~~~~
// CHECK:STDERR:
class package.C {
}

// CHECK:STDOUT: - filename: fail_in_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'package', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: 'package', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
