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
[[@LSP:bar:content]]
[[@LSP:baz:
multi
line
]]
[[@LSP-CALL:bar:content]]
[[@LSP-CALL:baz:
multi
line]]
[[@LSP-REPLY:7]]
[[@LSP-REPLY:8:bar]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: --- STDIN:
// CHECK:STDERR: Content-Length: 37
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 37
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 46
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "bar", content}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 50
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "baz",
// CHECK:STDERR: multi
// CHECK:STDERR: line
// CHECK:STDERR: }
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 67
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "bar", "id": 1, "params": {content}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 71
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "baz", "id": 2, "params": {
// CHECK:STDERR: multi
// CHECK:STDERR: line}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 31
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "7"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 48
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "8", "result": {bar}}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 38
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "exit"}
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDOUT: 1 args: `default_args`
