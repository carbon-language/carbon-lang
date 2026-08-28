// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_empty.carbon

// This test must be in a file by itself because it produces a fatal clang error
// (file not found), which suppresses all later clang errors in the same
// compilation.

// --- fail_import_cpp_library_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_empty.carbon:[[@LINE+2]]:10-11: error: empty filename [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "";
