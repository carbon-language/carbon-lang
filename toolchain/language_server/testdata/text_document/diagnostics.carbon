// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/text_document/diagnostics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/text_document/diagnostics.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/test.carbon", "languageId": "carbon",
                   "version": 1, "text": "{"}
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon", "version": 2},
  "contentChanges": [{"text": "fn F() { return (); }"}]
]]
[[@LSP-NOTIFY:textDocument/didClose:
  "textDocument": {"uri": "file:/test.carbon"}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

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
// CHECK:STDOUT: }Content-Length: 1164{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "opening symbol without a corresponding closing symbol",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 2,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 0,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "unrecognized declaration introducer",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 2,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 0,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "semantics TODO: `handle invalid parse trees in `check``",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 2,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 0,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///test.carbon",
// CHECK:STDOUT:     "version": 1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 507{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "no return expression should be provided in this context",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 20,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 9,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///test.carbon",
// CHECK:STDOUT:     "version": 2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 144{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///test.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
