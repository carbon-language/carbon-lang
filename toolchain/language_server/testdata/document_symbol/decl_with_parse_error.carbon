// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/document_symbol/decl_with_parse_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/document_symbol/decl_with_parse_error.carbon

// A declaration with a parse error has its node token set to wherever the
// parser gave up, which here is the `fn` introducer, before the name. The
// symbol's `range` must still contain its `selectionRange`, or the client
// rejects the whole response and no symbols are shown at all.

// --- decl_with_parse_error.carbon
fn f

fn G() {}

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/decl_with_parse_error.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/decl_with_parse_error.carbon"}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// CHECK:STDOUT: Content-Length: 394{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "capabilities": {
// CHECK:STDOUT:       "declarationProvider": true,
// CHECK:STDOUT:       "definitionProvider": true,
// CHECK:STDOUT:       "documentFormattingProvider": true,
// CHECK:STDOUT:       "documentSymbolProvider": true,
// CHECK:STDOUT:       "hoverProvider": true,
// CHECK:STDOUT:       "positionEncoding": "utf-16",
// CHECK:STDOUT:       "referencesProvider": true,
// CHECK:STDOUT:       "textDocumentSync": 2,
// CHECK:STDOUT:       "typeDefinitionProvider": true
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 877{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "semantics TODO: `handle invalid parse trees in `check``",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 4,
// CHECK:STDOUT:             "line": 0
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
// CHECK:STDOUT:         "message": "`fn` declarations must either end with a `;` or have a `{ ... }` block for a definition",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 2,
// CHECK:STDOUT:             "line": 2
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 0,
// CHECK:STDOUT:             "line": 2
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///decl_with_parse_error.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 867{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "kind": 12,
// CHECK:STDOUT:       "name": "f",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 4,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 4,
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
// CHECK:STDOUT:       "name": "G",
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 9,
// CHECK:STDOUT:           "line": 2
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 0,
// CHECK:STDOUT:           "line": 2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "selectionRange": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 4,
// CHECK:STDOUT:           "line": 2
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 3,
// CHECK:STDOUT:           "line": 2
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
