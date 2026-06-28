// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind format --output=- --lines=2:2 %s
//
// Formatting only the `fn F() {` line (line 2; line 1 is the blank above the
// code) reformats its matching `}` too, fixing the dangling indentation even
// though the `}` line is outside the requested range.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/range/brace_expansion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/range/brace_expansion.carbon

// --- test.carbon

fn F() {
  var x: i32 = 1;
      }

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT:
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var x: i32 = 1;
// CHECK:STDOUT: }
