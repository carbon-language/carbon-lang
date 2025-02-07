// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/document_symbol/language.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/document_symbol/language.carbon

// --- STDIN
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/test.cpp"}
]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: /test.cpp: warning: non-Carbon file requested [LanguageServerFileUnsupported]
// CHECK:STDERR:
// CHECK:STDOUT:
