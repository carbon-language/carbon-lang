// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_omit_library_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_omit_library_keyword.carbon

// CHECK:STDERR: fail_omit_library_keyword.carbon:[[@LINE+2]]:17-24: error: missing `library` keyword [MissingLibraryKeyword]
// CHECK:STDERR:
import Geometry "Shapes";

// CHECK:STDOUT: - filename: fail_omit_library_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Geometry'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
