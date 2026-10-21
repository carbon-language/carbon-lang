// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/basics/quiet_notifications.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/basics/quiet_notifications.carbon

// Notifications that we accept without doing anything, and so should produce no
// output at all. `$/` notifications are ignored whether or not we know them.

// --- STDIN
[[@LSP-NOTIFY:initialized]]
[[@LSP-NOTIFY:textDocument/didSave:
  "textDocument": {"uri": "file:/unknown.carbon"}
]]
[[@LSP-NOTIFY:$/cancelRequest:"id": 1]]
[[@LSP-NOTIFY:$/unknown-notify]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT:
