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

// CHECK:STDERR:        ╭─┤ fail_identifier.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ library Shapes;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: expected `default` or a string literal to specify the library
// CHECK:STDERR:        ·           ┃        name
// CHECK:STDERR:        ·           ┃        [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library Shapes;

// --- fail_raw_identifier.carbon

// CHECK:STDERR:        ╭─┤ fail_raw_identifier.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ library r#default;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: expected `default` or a string literal to specify the library
// CHECK:STDERR:        ·           ┃        name
// CHECK:STDERR:        ·           ┃        [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library r#default;

// --- fail_missing.carbon

// CHECK:STDERR:        ╭─┤ fail_missing.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ library;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: expected `default` or a string literal to specify the library
// CHECK:STDERR:        ·           ┃        name
// CHECK:STDERR:        ·           ┃        [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library;

// --- fail_missing_with_api.carbon

// CHECK:STDERR:        ╭─┤ fail_missing_with_api.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ library;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: expected `default` or a string literal to specify the library
// CHECK:STDERR:        ·           ┃        name
// CHECK:STDERR:        ·           ┃        [ExpectedLibraryNameOrDefault]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library;

// --- fail_trailing_impl.carbon

// CHECK:STDERR:        ╭─┤ fail_trailing_impl.carbon:[[@LINE+7]]:24
// CHECK:STDERR: ->     │ library "trailing_impl" impl
// CHECK:STDERR:        ·                        ┳
// CHECK:STDERR:        ·                        ┗━━┫ error: `library` declarations must end with a `;`
// CHECK:STDERR:        ·                           ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library "[[@TEST_NAME]]" impl

// --- fail_no_semi.carbon

// CHECK:STDERR:        ╭─┤ fail_no_semi.carbon:[[@LINE+7]]:18
// CHECK:STDERR: ->     │ library "no_semi"
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: `library` declarations must end with a `;`
// CHECK:STDERR:        ·                     ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library "[[@TEST_NAME]]"

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
