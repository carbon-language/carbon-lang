// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/library/fail_invalid_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/library/fail_invalid_name.carbon

// --- fail_identifier.carbon

// CHECK:STDERR: fail_identifier.carbon:[[@LINE+2]]:9-14: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:
library Shapes;

// --- fail_raw_identifier.carbon

// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE+2]]:9-17: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:
library r#default;

// --- fail_missing.carbon

// CHECK:STDERR: fail_missing.carbon:[[@LINE+2]]:8: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:
library;

// --- fail_missing_with_api.carbon

// CHECK:STDERR: fail_missing_with_api.carbon:[[@LINE+2]]:8: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:
library;

// --- fail_trailing_impl.carbon

// CHECK:STDERR: fail_trailing_impl.carbon:[[@LINE+2]]:25-28: error: `library` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
library "[[@TEST_NAME]]" impl

// --- fail_no_semi.carbon

library "[[@TEST_NAME]]"

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+39]]:26: error: `library` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_raw_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_with_api.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_trailing_impl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"trailing_impl"'
// CHECK:STDOUT:   ├─LibraryDecl 'impl' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"no_semi"'
// CHECK:STDOUT:   ├─LibraryDecl '"no_semi"' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
