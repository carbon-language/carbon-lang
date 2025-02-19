// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/basics/notify_parse_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/basics/notify_parse_error.carbon


// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:"nope": "bad"]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: warning: -32602: in call to `textDocument/didOpen`, JSON parse failed: missing value at (root).textDocument [LanguageServerNotificationParseError]
// CHECK:STDERR:
// CHECK:STDOUT: Content-Length: 146{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "capabilities": {
// CHECK:STDOUT:       "documentSymbolProvider": true,
// CHECK:STDOUT:       "textDocumentSync": 2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
