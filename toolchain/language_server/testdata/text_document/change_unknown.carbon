// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/text_document/change_unknown.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/text_document/change_unknown.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": [{"text": "new content"}]
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: /test.carbon: warning: unknown file requested [LanguageServerFileUnknown]
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
