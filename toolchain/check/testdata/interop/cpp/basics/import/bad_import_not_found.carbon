// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_not_found.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import_not_found.carbon

// This test must be in a file by itself because it produces a fatal clang error
// (file not found), which suppresses all later clang errors in the same
// compilation.

// --- fail_cpp_file_not_found.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_cpp_file_not_found.carbon:[[@LINE+6]]:10
// CHECK:STDERR: ->     │ #include "not_found.h"
// CHECK:STDERR:        ·          ━━━━━━┳━━━━━━
// CHECK:STDERR:        ·                ┗━━┫ error: 'not_found.h' file not found [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library "not_found.h";
