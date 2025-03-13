// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --fuzzing compile --no-prelude-import --phase=check %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/fail_fuzzing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/fail_fuzzing.carbon

// CHECK:STDERR: fail_fuzzing.carbon:[[@LINE+4]]:1: error: `Cpp` import found during fuzzing [CppInteropFuzzing]
// CHECK:STDERR: import Cpp library "file.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "file.h";
