// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/cpp_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/cpp_inline.carbon

// --- fail_bad_syntax.carbon

// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:18: error: expected string literal after `inline` [ExpectedStringAfterInline]
// CHECK:STDERR: import Cpp inline;
// CHECK:STDERR:                  ^
// CHECK:STDERR:
import Cpp inline;
// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:19: error: expected string literal after `inline` [ExpectedStringAfterInline]
// CHECK:STDERR: import Cpp inline library "foo.h";
// CHECK:STDERR:                   ^~~~~~~
// CHECK:STDERR:
import Cpp inline library "foo.h";

// TODO: We might want to allow this as a "preprocessed source" syntax.
// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:28: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: import Cpp library "foo.h" inline "bar";
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR:
import Cpp library "foo.h" inline "bar";

// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:24: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: import library "foo.h" inline "bar";
// CHECK:STDERR:                        ^~~~~~
// CHECK:STDERR:
import library "foo.h" inline "bar";

// --- fail_no_semi_at_eof.carbon

import Cpp inline
// CHECK:STDERR: fail_no_semi_at_eof.carbon:[[@LINE+4]]:1: error: expected string literal after `inline` [ExpectedStringAfterInline]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_package_inline.carbon

// CHECK:STDERR: fail_package_inline.carbon:[[@LINE+4]]:16: error: `package` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: package NotCpp inline "int n;";
// CHECK:STDERR:                ^~~~~~
// CHECK:STDERR:
package NotCpp inline "int n;";

// --- fail_library_inline.carbon

// CHECK:STDERR: fail_library_inline.carbon:[[@LINE+4]]:17: error: `library` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: library "foo.h" inline "int n;";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
library "foo.h" inline "int n;";

// --- wrong_package.carbon

// We reject this in check.
import NotCpp inline "bar";

// --- valid_syntax.carbon

import Cpp inline "int m;";

import Cpp inline '''
// C++ comment.
int n;
''';

// CHECK:STDOUT: - filename: fail_bad_syntax.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: 'library', has_error: yes},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"foo.h"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"foo.h"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_semi_at_eof.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: '', has_error: yes},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: 'inline', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_package_inline.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'NotCpp'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_library_inline.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"foo.h"'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: wrong_package.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'NotCpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: '"bar"'},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: valid_syntax.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: '"int m;"'},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'CppPackageName', text: 'Cpp'},
// CHECK:STDOUT:         {kind: 'InlineImportBody', text: ''''
// CHECK:STDOUT: // C++ comment.
// CHECK:STDOUT: int n;
// CHECK:STDOUT: ''''},
// CHECK:STDOUT:       {kind: 'InlineImportSpecifier', text: 'inline', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
