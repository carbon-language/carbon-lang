// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile %s
//
// AUTOUPDATE

fn F() {
  // Create diagnostics containing string references, and trigger reallocation
  // of the string table.
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+3]]:3: ERROR: Name `undeclared1` not found.
  // CHECK:STDERR:   undeclared1;
  // CHECK:STDERR:   ^~~~~~~~~~~
  undeclared1;

  // Add the name into the string table from the tokenized buffer's string
  // literal storage. Use a hex escape to ensure that the tokenized buffer
  // allocates separate storage for the result.
  "undec\x6Cared2";
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+3]]:3: ERROR: Name `undeclared2` not found.
  // CHECK:STDERR:   undeclared2;
  // CHECK:STDERR:   ^~~~~~~~~~~
  undeclared2;

  // Add the name into the string table via a declaration rather than an expression.
  if (true) { var undeclared3: i32 = 0; }
  // CHECK:STDERR: fail_flush_errors.carbon:[[@LINE+3]]:3: ERROR: Name `undeclared3` not found.
  // CHECK:STDERR:   undeclared3;
  // CHECK:STDERR:   ^~~~~~~~~~~
  undeclared3;
}
