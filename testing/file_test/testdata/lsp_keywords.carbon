// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/lsp_keywords.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/lsp_keywords.carbon

// --- STDIN
[[@LSP:foo:]]
[[@LSP:foo]]
[[@LSP:bar:"content": 0]]
[[@LSP:baz:
"multi": 0,
"line": 1
]]
[[@LSP-CALL:bar:"content": 0]]
[[@LSP-CALL:baz:
"multi": 0,
"line": 1]]
[[@LSP-REPLY:7]]
[[@LSP-REPLY:8:"bar": 0]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: --- STDIN:
// CHECK:STDERR: Content-Length: 34
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 34
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 46
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"bar","content":0}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 53
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"baz","multi":0,"line":1}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 64
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"bar","id":1,"params":{"content":0}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 71
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"baz","id":2,"params":{"line":1,"multi":0}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 28
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","id":"7"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 47
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","id":"8","result":{"bar":0}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 35
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"exit"}
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDOUT: 1 args: `default_args`
