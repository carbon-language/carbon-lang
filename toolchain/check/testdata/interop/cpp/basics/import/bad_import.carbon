// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_import.carbon

// --- fail_unterminated_import_inline.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
// CHECK:STDERR: <carbon Cpp imports>:1:2: error: expected '}' [CppInteropParseError]
// CHECK:STDERR:     1 | ;
// CHECK:STDERR:       |  ^
// CHECK:STDERR: fail_unterminated_import_inline.carbon:[[@LINE+4]]:10: note: to match this '{' [CppInteropParseNote]
// CHECK:STDERR:    12 | void f() {
// CHECK:STDERR:       |          ^
// CHECK:STDERR:
void f() {
''';

import Cpp inline '''c++
// CHECK:STDERR: fail_unterminated_import_inline.carbon:[[@LINE+4]]:1: error: extraneous closing brace ('}') [CppInteropParseError]
// CHECK:STDERR:    20 | }
// CHECK:STDERR:       | ^
// CHECK:STDERR:
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
