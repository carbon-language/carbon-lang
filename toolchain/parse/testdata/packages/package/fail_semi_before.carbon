// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/fail_semi_before.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/fail_semi_before.carbon

;

// CHECK:STDERR: fail_semi_before.carbon:[[@LINE+3]]:1-7: error: the `package` declaration must be the first non-comment line [PackageTooLate]
// CHECK:STDERR: fail_semi_before.carbon:[[@LINE-3]]:1: note: first non-comment line is here [FirstNonCommentLine]
// CHECK:STDERR:
package Geometry;

// CHECK:STDOUT: - filename: fail_semi_before.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   ├─EmptyDecl ';'
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   ├─PackageDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
