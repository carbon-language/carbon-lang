// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/formatting/range.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/formatting/range.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/range.carbon", "languageId": "carbon",
                   "text": "fn F() {\n  var x: i32 = 1+1;\n  var y: i32 = 2+2;\n}\n"}
]]
[[@LSP-CALL:textDocument/rangeFormatting:
  "textDocument": {"uri": "file:/range.carbon"},
  "range": {"start": {"line": 1, "character": 0},
            "end": {"line": 2, "character": 0}}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: Content-Length: 235{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "capabilities": {
// CHECK:STDOUT:       "documentFormattingProvider": true,
// CHECK:STDOUT:       "documentRangeFormattingProvider": true,
// CHECK:STDOUT:       "documentSymbolProvider": true,
// CHECK:STDOUT:       "textDocumentSync": 2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 757{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "binding `x` unused",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 12,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 6,
// CHECK:STDOUT:             "line": 1
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 2,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "binding `y` unused",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 12,
// CHECK:STDOUT:             "line": 2
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 6,
// CHECK:STDOUT:             "line": 2
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 2,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///range.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 473{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 16,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 16,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 17,
// CHECK:STDOUT:           "line": 1
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 17,
// CHECK:STDOUT:           "line": 1
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
