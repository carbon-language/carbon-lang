// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/fail_library_is_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/fail_library_is_identifier.carbon

// CHECK:STDERR: fail_library_is_identifier.carbon:[[@LINE+4]]:26: error: expected a string literal to specify the library name [ExpectedLibraryName]
// CHECK:STDERR: package Geometry library Shapes;
// CHECK:STDERR:                          ^~~~~~
// CHECK:STDERR:
package Geometry library Shapes;

// CHECK:STDOUT: - filename: fail_library_is_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Geometry'
// CHECK:STDOUT:   │ │ ╭─LibraryName 'Shapes' ❌
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─PackageDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
