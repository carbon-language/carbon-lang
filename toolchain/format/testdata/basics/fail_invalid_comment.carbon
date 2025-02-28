// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_invalid_comment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_invalid_comment.carbon

// --- fail_test.carbon

//f

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_test.carbon:2:3: error: whitespace is required after '//' [NoWhitespaceAfterCommentIntroducer]
// CHECK:STDERR: //f
// CHECK:STDERR:   ^
// CHECK:STDERR:
// CHECK:STDOUT:
// CHECK:STDOUT: //f
// CHECK:STDOUT:
