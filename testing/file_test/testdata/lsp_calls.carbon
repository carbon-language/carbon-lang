// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/lsp_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/lsp_calls.carbon

// --- STDIN
[[@LSP:foo:]]
[[@LSP:foo]]
[[@LSP:bar:content]]
[[@LSP:baz:
multi
line
]]
[[@LSP-REPLY:7]]
[[@LSP-REPLY:8:bar]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: --- STDIN:
// CHECK:STDERR: Content-Length: 48
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "1", "method": "foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 48
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "2", "method": "foo"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 57
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "3", "method": "bar", content}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 61
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "4", "method": "baz",
// CHECK:STDERR: multi
// CHECK:STDERR: line
// CHECK:STDERR: }
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 31
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "7"}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 36
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "id": "8", bar}
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 38
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc": "2.0", "method": "exit"}
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDOUT: 1 args: `default_args`
