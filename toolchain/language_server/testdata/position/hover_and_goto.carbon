// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/position/hover_and_goto.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/position/hover_and_goto.carbon

// Requests are answered at the name a user points at, not at the token the
// parser happened to finish a declaration on, so `definition` from the call on
// line 4 lands on `Abs` on line 0 rather than on the `{` of its body.
// `references` matches on that same name, which is why asking at the
// declaration of `x` still finds its use.

// --- position.carbon
fn Abs(n: i32) -> i32 { return n; }

fn Run() {
  var x: i32 = 1;
  Abs(x);
}

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/position.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 4, "character": 2}
]]
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 4, "character": 2}
]]
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 4, "character": 6}
]]
[[@LSP-CALL:textDocument/references:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 3, "character": 6},
  "context": {"includeDeclaration": true}
]]
[[@LSP-CALL:textDocument/typeDefinition:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 4, "character": 6}
]]
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 2, "character": 0}
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
// CHECK:STDOUT: }Content-Length: 1570{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "`Core.Int` implicitly referenced here, but package `Core` not found",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 13,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 10,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "`Core.Int` implicitly referenced here, but package `Core` not found",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 21,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 18,
// CHECK:STDOUT:             "line": 0
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "`Core.Int` implicitly referenced here, but package `Core` not found",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 12,
// CHECK:STDOUT:             "line": 3
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 9,
// CHECK:STDOUT:             "line": 3
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       },
// CHECK:STDOUT:       {
// CHECK:STDOUT:         "message": "`Core.Destroy` implicitly referenced here, but package `Core` not found",
// CHECK:STDOUT:         "range": {
// CHECK:STDOUT:           "end": {
// CHECK:STDOUT:             "character": 12,
// CHECK:STDOUT:             "line": 3
// CHECK:STDOUT:           },
// CHECK:STDOUT:           "start": {
// CHECK:STDOUT:             "character": 2,
// CHECK:STDOUT:             "line": 3
// CHECK:STDOUT:           }
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "severity": 1,
// CHECK:STDOUT:         "source": "carbon"
// CHECK:STDOUT:       }
// CHECK:STDOUT:     ],
// CHECK:STDOUT:     "uri": "file:///position.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 309{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```carbon\nAbs: <type of Abs>\n```"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 5,
// CHECK:STDOUT:         "line": 4
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 2,
// CHECK:STDOUT:         "line": 4
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 278{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 3,
// CHECK:STDOUT:           "line": 0
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///position.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 278{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 4,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 7,
// CHECK:STDOUT:           "line": 3
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 3
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///position.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 505{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 5,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 7,
// CHECK:STDOUT:           "line": 3
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 3
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///position.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 7,
// CHECK:STDOUT:           "line": 4
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 4
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///position.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 49{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 6,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": []
// CHECK:STDOUT: }Content-Length: 73{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 7,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": null
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 51{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 8,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
