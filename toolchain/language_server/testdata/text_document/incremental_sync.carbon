// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/text_document/incremental_sync.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/text_document/incremental_sync.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/test.carbon", "languageId": "carbon",
                   "text": "fn F() {}"}
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": [
    {"range": {"end": {"character": 4, "line": 0},
                 "start": { "character": 3, "line": 0 }},
      "text": "G" }
  ]
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": [
    { "range": { "end": { "character": 4, "line": 0 },
                 "start": { "character": 3, "line": 0 }},
      "text": "test" },
    { "range": { "end": { "character": 12, "line": 0 },
                 "start": { "character": 12, "line": 0 }},
      "text": "\nfn H() {}\n" },
    { "range": { "end": { "character": 4, "line": 0 },
                 "start": { "character": 3, "line": 0 }},
      "text": "T" }
  ]
]]
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/test.carbon"}
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": [
    { "text": "fn Updated() {}" }
  ]
]]
[[@LSP-CALL:textDocument/documentSymbol:
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
// CHECK:STDOUT: }Content-Length: 144{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///test.carbon"
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
// CHECK:STDOUT: }Content-Length: 144{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///test.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 871{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "kind": 12,
// CHECK:STDOUT:       "name": "Test",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 12,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 7,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 3,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "kind": 12,
// CHECK:STDOUT:       "name": "H",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 9,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 4,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 3,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 144{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///test.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 467{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "kind": 12,
// CHECK:STDOUT:       "name": "Updated",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 15,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 10,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 3,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 4,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
