// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind link foo.o bar.o
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/link/fail_output_missing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/link/fail_output_missing.carbon
// CHECK:STDERR: error: no output specified to a link command and no extra Clang options that can provide an output [LinkOutputOptionMissing]
// CHECK:STDERR:

// --- foo.carbon
