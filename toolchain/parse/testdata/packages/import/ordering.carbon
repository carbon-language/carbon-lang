// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/ordering.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/ordering.carbon

// --- after_import.carbon

import A;
import B;

// --- after_library.carbon

library "[[@TEST_NAME]]";

import B;

// --- after_package.carbon

package A;

import B;

// CHECK:STDOUT: - filename: after_import.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'A'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: after_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"after_library"'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: after_package.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'A'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
