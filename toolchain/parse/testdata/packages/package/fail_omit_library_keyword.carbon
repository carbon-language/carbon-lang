// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/fail_omit_library_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/fail_omit_library_keyword.carbon

// CHECK:STDERR: fail_omit_library_keyword.carbon:[[@LINE+2]]:18-25: error: missing `library` keyword [MissingLibraryKeyword]
// CHECK:STDERR:
package Geometry "Shapes";

// CHECK:STDOUT: - filename: fail_omit_library_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Geometry'
// CHECK:STDOUT:   ├─PackageDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
