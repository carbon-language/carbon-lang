// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: At present, this has a
// "dropping diagnostic in /test.carbon.generated.cpp_imports.h". That should be
// fixed to assign back to this file, but that may come from other fixes to C++
// diagnostic output.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/text_document/open_with_cpp_nonexistent.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/text_document/open_with_cpp_nonexistent.carbon

// --- STDIN
[[@LSP-CALL:initialize]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/test.carbon", "languageId": "carbon",
                   "text": "import Cpp library \"nonexistent.h\";"}
]]
[[@LSP-NOTIFY:textDocument/didClose:
  "textDocument": {"uri": "file:/test.carbon"}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: /test.carbon: warning: dropping diagnostic in /test.carbon.generated.cpp_imports.h:
// CHECK:STDERR: /test.carbon.generated.cpp_imports.h:1: error: C++:
// CHECK:STDERR: /test.carbon.generated.cpp_imports.h:1:10: fatal error: 'nonexistent.h' file not found
// CHECK:STDERR:     1 | #include "nonexistent.h"
// CHECK:STDERR:       |          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /test.carbon:1:1: note: in `Cpp` import
// CHECK:STDERR: import Cpp library "nonexistent.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:  [LanguageServerDiagnosticInWrongFile]
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
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
