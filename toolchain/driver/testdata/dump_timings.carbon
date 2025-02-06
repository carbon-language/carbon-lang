// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --dump-timings --phase=check %s
//
// SET-CHECK-SUBSET
//
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/dump_timings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/dump_timings.carbon

// CHECK:STDOUT: filename:        dump_timings.carbon
// CHECK:STDOUT: nanoseconds:
// CHECK:STDOUT:   lex:             {{\d+}}
// CHECK:STDOUT:   parse:           {{\d+}}
// CHECK:STDOUT:   check:           {{\d+}}
// CHECK:STDOUT:   Total:           {{\d+}}
// CHECK:STDOUT: ...
