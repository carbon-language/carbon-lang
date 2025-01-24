// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/unsupported_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/unsupported_call.carbon


// --- STDIN
[[@LSP:unknown-call]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: Content-Length: 122{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "error": {
// CHECK:STDOUT:     "code": -32601,
// CHECK:STDOUT:     "message": "unsupported call `unknown-call`"
// CHECK:STDOUT:   },
// CHECK:STDOUT:   "id": "1",
// CHECK:STDOUT:   "jsonrpc": "2.0"
// CHECK:STDOUT: }
