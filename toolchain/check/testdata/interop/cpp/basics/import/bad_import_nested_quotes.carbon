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
// CHECK:STDERR:        ╭─┤ fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+23]]:10
// CHECK:STDERR: ->     │ #include "\"foo.h\""
// CHECK:STDERR:        ·          ━┳━
// CHECK:STDERR:        ·           ┗━━┫ error: '\' file not found [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+17]]:13
// CHECK:STDERR: ->     │ #include "\"foo.h\""
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: invalid suffix on literal; C++11 requires a space
// CHECK:STDERR:        ·             │  ┃        between literal and identifier
// CHECK:STDERR:        ·             │  ┃        [CppInteropParseError]
// CHECK:STDERR:        ·             ╰────┤ insert ` ` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_import_cpp_library_file_with_quotes.carbon:[[@LINE+8]]:13
// CHECK:STDERR: ->     │ #include "\"foo.h\""
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ warning: extra tokens at end of #include directive
// CHECK:STDERR:        ·             │  ┃          [CppInteropParseWarning]
// CHECK:STDERR:        ·             ╰────┤ insert `//` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library "\"foo.h\"";
