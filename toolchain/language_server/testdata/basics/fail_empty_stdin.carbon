// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/basics/fail_empty_stdin.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/basics/fail_empty_stdin.carbon

// --- STDIN
// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: error: Input/output error [LanguageServerTransportError]
// CHECK:STDERR:
// CHECK:STDOUT:
