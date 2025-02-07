// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/text_document/change_count.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/text_document/change_count.carbon

// --- STDIN
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {"uri": "file:/test.carbon", "languageId": "carbon",
                   "text": "test"}
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": []
]]
[[@LSP-NOTIFY:textDocument/didChange:
  "textDocument": {"uri": "file:/test.carbon"},
  "contentChanges": [
    {"range": {"start": {"line": 5, "character": 23 },
               "end": {"line": 6, "character": 0 }},
     "text": "a"},
    {"range": {"start": {"line": 5, "character": 23 },
               "end": {"line": 6, "character": 0 }},
     "text": "a"}
  ]
]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: /test.carbon: warning: received unsupported contentChanges count: 0 [LanguageServerUnsupportedChanges]
// CHECK:STDERR:
// CHECK:STDERR: /test.carbon: warning: received unsupported contentChanges count: 2 [LanguageServerUnsupportedChanges]
// CHECK:STDERR:
// CHECK:STDOUT:
