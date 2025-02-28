// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_library_is_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_library_is_identifier.carbon

// CHECK:STDERR: fail_library_is_identifier.carbon:[[@LINE+4]]:25: error: expected a string literal to specify the library name [ExpectedLibraryName]
// CHECK:STDERR: import Geometry library Shapes;
// CHECK:STDERR:                         ^~~~~~
// CHECK:STDERR:
import Geometry library Shapes;

// CHECK:STDOUT: - filename: fail_library_is_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Geometry'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: 'Shapes', has_error: yes},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
