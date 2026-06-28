// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_best_effort.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_best_effort.carbon

// Even with a parse error (here, a missing `;`), formatting is best-effort: it
// still normalizes whitespace and produces output rather than giving up.

// --- fail_missing_semicolon.carbon

fn  F ( )  {
  bar
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_missing_semicolon.carbon:4:1: error: expected `;` after expression statement [ExpectedExprSemi]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   bar
// CHECK:STDOUT: }
