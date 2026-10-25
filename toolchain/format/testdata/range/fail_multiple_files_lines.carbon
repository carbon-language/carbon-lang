// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind format --output=- --lines=1:1 foo.carbon bar.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/range/fail_multiple_files_lines.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/range/fail_multiple_files_lines.carbon
// CHECK:STDERR: error: multiple input files provided; `--lines` only works with one input file [FormatMultipleFilesWithLines]
// CHECK:STDERR:

// --- foo.carbon
fn F() {}

// --- bar.carbon
fn G() {}
