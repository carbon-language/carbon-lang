// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: -v language-server
//
// Verbose output includes a timestamp.
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/verbose.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/verbose.carbon

// CHECK:STDERR: V[{{.+}}] <<< {
// CHECK:STDERR:   "jsonrpc": "2.0",
// CHECK:STDERR:   "method": "exit"
// CHECK:STDERR: }
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDOUT:

// --- STDIN
[[@LSP-NOTIFY:exit:]]
