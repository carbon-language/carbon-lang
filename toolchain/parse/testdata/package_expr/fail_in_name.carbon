// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+4]]:14: error: unexpected expression before : in binding pattern [ExpectedBindingName]
// CHECK:STDERR: var package.n: i32;
// CHECK:STDERR:              ^
// CHECK:STDERR:
var package.n: i32;

// `val` is a keyword, but during error recovery we treat it as an identifier.
// CHECK:STDERR: fail_in_name.carbon:[[@LINE+8]]:13: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: var package.val: i32;
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_in_name.carbon:[[@LINE+4]]:16: error: unexpected expression before : in binding pattern [ExpectedBindingName]
// CHECK:STDERR: var package.val: i32;
// CHECK:STDERR:                ^
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
// CHECK:STDOUT:           {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'n'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'val'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
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
