// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_flush_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_flush_errors.carbon

fn F() {
  // Create diagnostics containing string references, and trigger reallocation
  // of the string table.
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+4]]:3: error: name `undeclared1` not found [NameNotFound]
  // CHECK:STDERR:   undeclared1;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  undeclared1;

  // Add the name into the string table from the tokenized buffer's string
  // literal storage. Use a hex escape to ensure that the tokenized buffer
  // allocates separate storage for the result.
  "undec\x6Cared2";
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+4]]:3: error: name `undeclared2` not found [NameNotFound]
  // CHECK:STDERR:   undeclared2;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  undeclared2;

  // Add the name into the string table via a declaration rather than an expression.
  if (true) { var undeclared3: () = (); }
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+4]]:3: error: name `undeclared3` not found [NameNotFound]
  // CHECK:STDERR:   undeclared3;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  undeclared3;
}
