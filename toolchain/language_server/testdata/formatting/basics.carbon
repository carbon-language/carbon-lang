// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/formatting/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/formatting/basics.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/messy.carbon", "languageId": "carbon",
                   "text": "fn  F ( x : i32 )->i32{return x ;}"}
]]
[[@LSP-CALL:textDocument/formatting:
  "textDocument": {"uri": "file:/messy.carbon"}
]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/formatted.carbon", "languageId": "carbon",
                   "text": "fn F() {}\n"}
]]
[[@LSP-CALL:textDocument/formatting:
  "textDocument": {"uri": "file:/formatted.carbon"}
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
// CHECK:STDOUT: }Content-Length: 145{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///messy.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 2576{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 4,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 2,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 5,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 8,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 7,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 10,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 9,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 16,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 15,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 17,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 17,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": " ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 22,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 22,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "\n  ",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 23,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 23,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 32,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 31,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "\n",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 33,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 33,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "newText": "\n",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 34,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 34,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 149{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///formatted.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 49{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": []
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 4,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
