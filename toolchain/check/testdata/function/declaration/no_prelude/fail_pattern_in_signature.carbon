// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_pattern_in_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_pattern_in_signature.carbon

// CHECK:STDERR: fail_pattern_in_signature.carbon:[[@LINE+8]]:6: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn F((a: {}, b: {}), c: {});
// CHECK:STDERR:      ^
// CHECK:STDERR:
// CHECK:STDERR: fail_pattern_in_signature.carbon:[[@LINE+4]]:6: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn F((a: {}, b: {}), c: {});
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F((a: {}, b: {}), c: {});

// CHECK:STDOUT: --- fail_pattern_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
