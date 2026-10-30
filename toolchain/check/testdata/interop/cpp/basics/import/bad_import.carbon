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
// CHECK:STDERR:        ╭─┤ <carbon Cpp imports>:1:2
// CHECK:STDERR: ->     │ ;
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: expected '}' [CppInteropParseError]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_unterminated_import_inline.carbon
// CHECK:STDERR:        │ void f() {
// CHECK:STDERR:        ·          ┬
// CHECK:STDERR:        ·          ╰──┤ to match this '{' [CppInteropParseNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:

// --- fail_unterminated_import_inline.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
void f() {
''';

import Cpp inline '''c++
// CHECK:STDERR:        ╭─┤ fail_unterminated_import_inline.carbon:[[@LINE+6]]:1
// CHECK:STDERR: ->     │ }
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: extraneous closing brace ('}') [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
}
''';

// --- fail_import_cpp_library_without_name_and_inline_without_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_without_name_and_inline_without_content.carbon:[[@LINE+16]]:20
// CHECK:STDERR: ->     │ import Cpp library inline;
// CHECK:STDERR:        ·                    ┳━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: expected a string literal to specify the library
// CHECK:STDERR:        ·                       ┃        name
// CHECK:STDERR:        ·                       ┃        [ExpectedLibraryName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_without_name_and_inline_without_content.carbon:[[@LINE+8]]:20
// CHECK:STDERR: ->     │ import Cpp library inline;
// CHECK:STDERR:        ·                    ┳━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: semantics TODO:
// CHECK:STDERR:        ·                       ┃        `handle invalid parse trees in `check``
// CHECK:STDERR:        ·                       ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library inline;

// --- fail_import_cpp_library_with_name_and_inline_without_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_with_name_and_inline_without_content.carbon:[[@LINE+14]]:28
// CHECK:STDERR: ->     │ import Cpp library "foo.h" inline;
// CHECK:STDERR:        ·                            ┳━━━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: `import` declarations must end with a `;`
// CHECK:STDERR:        ·                               ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_with_name_and_inline_without_content.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ import Cpp library "foo.h" inline;
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·                 ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library "foo.h" inline;

// --- fail_import_cpp_library_without_name_and_inline_with_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_without_name_and_inline_with_content.carbon:[[@LINE+16]]:20
// CHECK:STDERR: ->     │ import Cpp library inline "";
// CHECK:STDERR:        ·                    ┳━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: expected a string literal to specify the library
// CHECK:STDERR:        ·                       ┃        name
// CHECK:STDERR:        ·                       ┃        [ExpectedLibraryName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_without_name_and_inline_with_content.carbon:[[@LINE+8]]:20
// CHECK:STDERR: ->     │ import Cpp library inline "";
// CHECK:STDERR:        ·                    ┳━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: semantics TODO:
// CHECK:STDERR:        ·                       ┃        `handle invalid parse trees in `check``
// CHECK:STDERR:        ·                       ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library inline "";

// --- fail_import_cpp_library_with_name_and_inline_with_content.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_with_name_and_inline_with_content.carbon:[[@LINE+14]]:28
// CHECK:STDERR: ->     │ import Cpp library "foo.h" inline "";
// CHECK:STDERR:        ·                            ┳━━━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: `import` declarations must end with a `;`
// CHECK:STDERR:        ·                               ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_with_name_and_inline_with_content.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ import Cpp library "foo.h" inline "";
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·                 ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library "foo.h" inline "";
