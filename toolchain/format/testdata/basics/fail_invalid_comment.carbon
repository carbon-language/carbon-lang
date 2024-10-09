// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: This can't autoupdate because the STDERR is retained in test input.
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_invalid_comment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_invalid_comment.carbon

// CHECK:STDERR: fail_test.carbon:2:3: error: whitespace is required after '//'
// CHECK:STDERR: //f
// CHECK:STDERR:   ^
// CHECK:STDOUT:
// CHECK:STDOUT: //f

// --- fail_test.carbon

//f
