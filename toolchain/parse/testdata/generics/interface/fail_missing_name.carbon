// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_missing_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_missing_name.carbon

// CHECK:STDERR: fail_missing_name.carbon:[[@LINE+4]]:11: error: `interface` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: interface {
// CHECK:STDERR:           ^
// CHECK:STDERR:
interface {
}

// CHECK:STDOUT: - filename: fail_missing_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │ ├─InvalidParse '{' has_error
// CHECK:STDOUT:   ├─InterfaceDecl '}' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
