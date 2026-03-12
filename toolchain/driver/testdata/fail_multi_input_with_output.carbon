// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Exclude the prelude import so that the number of prelude files isn't tested.
// ARGS: --include-diagnostic-kind compile --output=foo.o --no-prelude-import foo.carbon bar.carbon
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_multi_input_with_output.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_multi_input_with_output.carbon
// CHECK:STDERR: error: writing 2 input files to the same `--output` file would overwrite the output file; for now, pass `--output-last-input-only` if intended [CompileMultipleInputsWithOutput]
// CHECK:STDERR:
