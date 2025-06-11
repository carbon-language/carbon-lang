// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-prelude-import
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/file_not_found.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/file_not_found.carbon

// --- fail_cpp_file_not_found.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_cpp_file_not_found.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: fail_cpp_file_not_found.carbon.generated.cpp_imports.h:1: error: fail_cpp_file_not_found.carbon.generated.cpp_imports.h:1:10: fatal error: 'not_found.h' file not found
// CHECK:STDERR:     1 | #include "not_found.h"
// CHECK:STDERR:       |          ^~~~~~~~~~~~~
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "not_found.h";
