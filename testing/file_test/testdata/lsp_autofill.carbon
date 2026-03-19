// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/lsp_autofill.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/lsp_autofill.carbon

// --- foo.carbon
class Foo {
  fn foo();
  fn bar() {}
}

// --- STDIN
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/foo.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: --- STDIN:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDERR: Content-Length: 204
// CHECK:STDERR:
// CHECK:STDERR: {"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"languageId":"carbon","text":"\n\n\n\n\n\n\n\n\n\n\nclass Foo {\n  fn foo();\n  fn bar() {}\n}\n\n","uri":"file:/foo.carbon"}}}
// CHECK:STDERR:
// CHECK:STDERR:
// CHECK:STDOUT: 2 args: `default_args`, `foo.carbon`
// CHECK:STDOUT: foo.carbon:12: class Foo {
// CHECK:STDOUT: foo.carbon:13:   fn foo();
// CHECK:STDOUT: foo.carbon:14:   fn bar() {}
// CHECK:STDOUT: foo.carbon:15: }
