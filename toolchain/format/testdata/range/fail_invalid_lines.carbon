// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind format --output=- --lines=5:2 %s
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/range/fail_invalid_lines.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/range/fail_invalid_lines.carbon
// CHECK:STDERR: error: invalid `--lines` value `5:2`; expected `START:END` with 1 <= START <= END [FormatInvalidLineRange]
// CHECK:STDERR:

// A reversed range (START > END) is rejected.

fn F() {}
