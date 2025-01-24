// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/parse_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/parse_error.carbon


// --- STDIN
[[@LSP-REPLY:1]]
[[@LSP-REPLY:2:"result": 3]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: warning: unexpected reply to request ID "1": null [LanguageServerUnexpectedReply]
// CHECK:STDERR:
// CHECK:STDERR: warning: unexpected reply to request ID "2": 3 [LanguageServerUnexpectedReply]
// CHECK:STDERR:
// CHECK:STDOUT:
