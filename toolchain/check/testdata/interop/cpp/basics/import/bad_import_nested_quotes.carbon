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

// --- fail_todo_import_cpp_library_file_with_quotes.carbon

library "[[@TEST_NAME]]";

// TODO: Translate the file to a `#include` that includes the named file
// properly. `\` escapes aren't processed in regular `#include`s.
// CHECK:STDERR: fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+8]]:10-12: error: '\' file not found [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+6]]:13-15: error: invalid suffix on literal; C++11 requires a space between literal and identifier [CppInteropParseError]
// CHECK:STDERR: fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+5]]:13: note: insert ` ` here [CppInteropFixItInsertion]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+3]]:13-15: warning: extra tokens at end of #include directive [CppInteropParseWarning]
// CHECK:STDERR: fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+2]]:13: note: insert `//` here [CppInteropFixItInsertion]
// CHECK:STDERR:
import Cpp library "\"foo.h\"";
