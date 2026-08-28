// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/document_symbol/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/document_symbol/incomplete.carbon

// --- incomplete.carbon
class Incomplete {
  fn Foo() {

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/incomplete.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/incomplete.carbon"}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// CHECK:STDOUT: Content-Length: 352{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "capabilities": {
// CHECK:STDOUT:       "declarationProvider": true,
// CHECK:STDOUT:       "definitionProvider": true,
// CHECK:STDOUT:       "documentSymbolProvider": true,
// CHECK:STDOUT:       "hoverProvider": true,
// CHECK:STDOUT:       "positionEncoding": "utf-16",
// CHECK:STDOUT:       "referencesProvider": true,
// CHECK:STDOUT:       "textDocumentSync": 2,
// CHECK:STDOUT:       "typeDefinitionProvider": true
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 834{{\r}}
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
// CHECK:STDOUT:             "character": 18,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 17,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "opening symbol without a corresponding closing symbol",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 12,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 11,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///incomplete.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 1004{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "children": [
// CHECK:STDOUT:         {
// CHECK:STDOUT:           "kind": 12,
// CHECK:STDOUT:           "name": "Foo",
// CHECK:STDOUT:           "range": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 13,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 2,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             }
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "selectionRange": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 8,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 5,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             }
// CHECK:STDOUT:           }
// CHECK:STDOUT:         }
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       "kind": 5,
// CHECK:STDOUT:       "name": "Incomplete",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 13,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 16,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
