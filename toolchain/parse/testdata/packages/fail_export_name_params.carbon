// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/fail_export_name_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/fail_export_name_params.carbon

// --- a.carbon

package Foo library "[[@TEST_NAME]]";

class C1(T:! type);
class C2(T:! type);

// --- fail_b.carbon

package Foo library "[[@TEST_NAME]]";

import library "a";

export C1;

// CHECK:STDERR: fail_b.carbon:[[@LINE+3]]:20: error: unexpected parameters after name declaration [UnexpectedParamsAfterDeclName]
// CHECK:STDERR: export C2(T:! type);
// CHECK:STDERR:                    ^
export C2(T:! type);

// CHECK:STDOUT: - filename: a.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"a"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'C1'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'C2'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_b.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"b"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"a"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'C1'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'C2'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
