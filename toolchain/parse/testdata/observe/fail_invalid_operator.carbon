// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/fail_invalid_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/fail_invalid_operator.carbon

// CHECK:STDERR: fail_invalid_operator.carbon:[[@LINE+2]]:11: error: observe should use `==` or `impls` operator [ExpectedObserveOperator]
// CHECK:STDERR:
observe T = U impls I;

// CHECK:STDOUT: - filename: fail_invalid_operator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   ├─ObserveDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
