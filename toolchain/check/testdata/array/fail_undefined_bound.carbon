// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_undefined_bound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_undefined_bound.carbon

// CHECK:STDERR: fail_undefined_bound.carbon:[[@LINE+4]]:8: error: semantics TODO: `HandleArrayExprWithoutBounds` [SemanticsTodo]
// CHECK:STDERR: var a: [i32; ];
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var a: [i32; ];

// CHECK:STDOUT: --- fail_undefined_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
