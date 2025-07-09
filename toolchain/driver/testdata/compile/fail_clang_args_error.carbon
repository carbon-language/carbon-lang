// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --clang-arg=-fclang-flag-that-doesnt-exist foo.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/fail_clang_args_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/fail_clang_args_error.carbon
// CHECK:STDERR: error: unknown argument: '-fclang-flag-that-doesnt-exist' [CppInteropDriverError]
// CHECK:STDERR:

// --- foo.carbon
