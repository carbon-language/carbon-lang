// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_as_declared_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_as_declared_name.carbon

namespace N;

// CHECK:STDERR: fail_base_as_declared_name.carbon:[[@LINE+8]]:6: error: `.` should be followed by a name [ExpectedDeclNameAfterPeriod]
// CHECK:STDERR: fn N.base() {}
// CHECK:STDERR:      ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_base_as_declared_name.carbon:[[@LINE+4]]:6: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn N.base() {}
// CHECK:STDERR:      ^~~~
// CHECK:STDERR:
fn N.base() {}

// CHECK:STDOUT: --- fail_base_as_declared_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
