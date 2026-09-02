// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/position/bad_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/position/bad_params.carbon
// CHECK:STDERR: /other_file.carbon: warning: unknown file requested [LanguageServerFileUnknown]
// CHECK:STDERR:
// CHECK:STDERR: /other_file.carbon: warning: unknown file requested [LanguageServerFileUnknown]
// CHECK:STDERR:
// CHECK:STDERR: /other_file.carbon: warning: unknown file requested [LanguageServerFileUnknown]
// CHECK:STDERR:
// CHECK:STDERR: /other_file.carbon: warning: unknown file requested [LanguageServerFileUnknown]
// CHECK:STDERR:
// CHECK:STDERR: /other_file.carbon: warning: unknown file requested [LanguageServerFileUnknown]
// CHECK:STDERR:

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/some_file.carbon",
    "languageId": "carbon",
    "text": ""
  }
]]
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/other_file.carbon"},
  "position": {"line": 4, "character": 2}
]]
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/other_file.carbon"},
  "position": {"line": 4, "character": 2}
]]
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/other_file.carbon"},
  "position": {"line": 4, "character": 6}
]]
[[@LSP-CALL:textDocument/references:
  "textDocument": {"uri": "file:/other_file.carbon"},
  "position": {"line": 3, "character": 6},
  "context": {"includeDeclaration": true}
]]
[[@LSP-CALL:textDocument/typeDefinition:
  "textDocument": {"uri": "file:/other_file.carbon"},
  "position": {"line": 4, "character": 6}
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
// CHECK:STDOUT: }Content-Length: 149{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///some_file.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 130{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32602,
// CHECK:STDOUT:     "message": "Unknown textDocument `/other_file.carbon`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }Content-Length: 130{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32602,
// CHECK:STDOUT:     "message": "Unknown textDocument `/other_file.carbon`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }Content-Length: 130{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32602,
// CHECK:STDOUT:     "message": "Unknown textDocument `/other_file.carbon`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": 4,
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }Content-Length: 130{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32602,
// CHECK:STDOUT:     "message": "Unknown textDocument `/other_file.carbon`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": 5,
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }Content-Length: 130{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32602,
// CHECK:STDOUT:     "message": "Unknown textDocument `/other_file.carbon`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": 6,
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 7,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
