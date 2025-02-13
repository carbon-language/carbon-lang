// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/fail_after_package.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/fail_after_package.carbon

package A;
// CHECK:STDERR: fail_after_package.carbon:[[@LINE+7]]:1: error: the `package` declaration must be the first non-comment line [PackageTooLate]
// CHECK:STDERR: package B;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_after_package.carbon:[[@LINE-4]]:1: note: first non-comment line is here [FirstNonCommentLine]
// CHECK:STDERR: package A;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package B;

// CHECK:STDOUT: - filename: fail_after_package.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'A'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
