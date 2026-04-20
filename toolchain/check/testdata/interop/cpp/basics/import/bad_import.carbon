// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/bad_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/bad_import.carbon

// --- fail_import_cpp_library_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_empty.carbon:[[@LINE+4]]:10: error: empty filename [CppInteropParseError]
// CHECK:STDERR:     8 | #include ""
// CHECK:STDERR:       |          ^
// CHECK:STDERR:
import Cpp library "";

// --- fail_import_cpp_library_file_with_quotes.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon:[[@LINE+4]]:10: error: '\"foo.h\"' file not found [CppInteropParseError]
// CHECK:STDERR:     8 | #include "\"foo.h\""
// CHECK:STDERR:       |          ^~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "\"foo.h\"";

// --- todo_fail_unterminated_import_inline.carbon

import Cpp inline '''c++
void f() {
''';

// TODO: We should diagnose that the inline C++ code didn't leave us at the top
// level.

import Cpp inline '''c++
}
''';

// --- fail_import_cpp_library_without_name_and_inline_without_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_without_name_and_inline_without_content.carbon:[[@LINE+8]]:20: error: expected a string literal to specify the library name [ExpectedLibraryName]
// CHECK:STDERR: import Cpp library inline;
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_without_name_and_inline_without_content.carbon:[[@LINE+4]]:20: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: import Cpp library inline;
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
import Cpp library inline;

// --- fail_import_cpp_library_with_name_and_inline_without_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_with_name_and_inline_without_content.carbon:[[@LINE+8]]:28: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: import Cpp library "foo.h" inline;
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_with_name_and_inline_without_content.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: import Cpp library "foo.h" inline;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "foo.h" inline;

// --- fail_import_cpp_library_without_name_and_inline_with_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_without_name_and_inline_with_content.carbon:[[@LINE+8]]:20: error: expected a string literal to specify the library name [ExpectedLibraryName]
// CHECK:STDERR: import Cpp library inline "";
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_without_name_and_inline_with_content.carbon:[[@LINE+4]]:20: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: import Cpp library inline "";
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
import Cpp library inline "";

// --- fail_import_cpp_library_with_name_and_inline_with_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_with_name_and_inline_with_content.carbon:[[@LINE+8]]:28: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: import Cpp library "foo.h" inline "";
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_with_name_and_inline_with_content.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: import Cpp library "foo.h" inline "";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "foo.h" inline "";
