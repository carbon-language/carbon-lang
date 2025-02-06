// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --phase=check --dump-mem-usage %s
//
// To test this file alone, run:
//   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/dump_mem_usage.carbon
// To dump output, run:
//   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/dump_mem_usage.carbon
// NOAUTOUPDATE
//
// Avoid testing specific values:
// SET-CHECK-SUBSET

var x: i32 = 1;

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        dump_mem_usage.carbon
// CHECK:STDOUT: source_:
// CHECK:STDOUT:   used_bytes:      0
// CHECK:STDOUT:   reserved_bytes:  0
// CHECK:STDOUT: ...
