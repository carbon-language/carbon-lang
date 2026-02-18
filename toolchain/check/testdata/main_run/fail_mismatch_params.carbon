// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/main_run/fail_mismatch_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/main_run/fail_mismatch_params.carbon

// --- fail_bad_argc.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_argc.carbon:[[@LINE+4]]:1: error: invalid parameters for `Main.Run` function; expected `()` or `(argc: i32, argv: Core.Optional(char*)*)` [InvalidMainRunParameters]
// CHECK:STDERR: fn Run(n: i64) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Run(n: i64) {}

// --- fail_bad_argv.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_argv.carbon:[[@LINE+4]]:1: error: invalid parameters for `Main.Run` function; expected `()` or `(argc: i32, argv: Core.Optional(char*)*)` [InvalidMainRunParameters]
// CHECK:STDERR: fn Run(n: i32, p: i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Run(n: i32, p: i32) {}

// --- fail_too_many_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many_params.carbon:[[@LINE+4]]:1: error: invalid parameters for `Main.Run` function; expected `()` or `(argc: i32, argv: Core.Optional(char*)*)` [InvalidMainRunParameters]
// CHECK:STDERR: fn Run(n: i32, p: i32*, q: i32*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Run(n: i32, p: i32*, q: i32*) {}
