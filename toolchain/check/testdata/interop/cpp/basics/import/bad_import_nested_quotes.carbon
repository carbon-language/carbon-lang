// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_nested_quotes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_nested_quotes.carbon

// This test must be in a file by itself because it produces a fatal clang error
// (file not found), which suppresses all later clang errors in the same
// compilation.

// --- fail_import_cpp_library_file_with_quotes.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon:[[@LINE+7]]:35: in expansion of macro defined here [InCppMacroExpansion]
// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon:[[@LINE+6]]:10: error: '\"foo.h\"' file not found [CppInteropParseError]
// CHECK:STDERR:    11 | #include __carbon_include_filename
// CHECK:STDERR:       |          ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:    11 | #define __carbon_include_filename "\"foo.h\""
// CHECK:STDERR:       |                                   ^~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "\"foo.h\"";
