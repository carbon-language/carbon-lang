// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/library/fail_too_late.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/library/fail_too_late.carbon

import Geometry;

// CHECK:STDERR: fail_too_late.carbon:[[@LINE+7]]:1: error: the `library` declaration must be the first non-comment line [PackageTooLate]
// CHECK:STDERR: library "too_late";
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_too_late.carbon:[[@LINE-5]]:1: note: first non-comment line is here [FirstNonCommentLine]
// CHECK:STDERR: import Geometry;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
library "[[@TEST_NAME]]";

// CHECK:STDOUT: - filename: fail_too_late.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Geometry'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
