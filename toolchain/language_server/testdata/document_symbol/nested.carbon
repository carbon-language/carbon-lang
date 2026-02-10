// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/document_symbol/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/document_symbol/nested.carbon

// --- class.carbon
class A {
  fn F();
  fn G() {}
}

// --- STDIN
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/class.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/class.carbon"}
]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: Content-Length: 145{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///class.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 1495{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "children": [
// CHECK:STDOUT:         {
// CHECK:STDOUT:           "kind": 12,
// CHECK:STDOUT:           "name": "F",
// CHECK:STDOUT:           "range": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 9,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 2,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             }
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "selectionRange": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 6,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 5,
// CHECK:STDOUT:               "line": 1
// CHECK:STDOUT:             }
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         {
// CHECK:STDOUT:           "kind": 12,
// CHECK:STDOUT:           "name": "G",
// CHECK:STDOUT:           "range": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 11,
// CHECK:STDOUT:               "line": 2
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 2,
// CHECK:STDOUT:               "line": 2
// CHECK:STDOUT:             }
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "selectionRange": {
// CHECK:STDOUT:             "end": {
// CHECK:STDOUT:               "character": 6,
// CHECK:STDOUT:               "line": 2
// CHECK:STDOUT:             },
// CHECK:STDOUT:             "start": {
// CHECK:STDOUT:               "character": 5,
// CHECK:STDOUT:               "line": 2
// CHECK:STDOUT:             }
// CHECK:STDOUT:           }
// CHECK:STDOUT:         }
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       "kind": 5,
// CHECK:STDOUT:       "name": "A",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 1,
// CHECK:STDOUT:           "line": 3
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
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }
