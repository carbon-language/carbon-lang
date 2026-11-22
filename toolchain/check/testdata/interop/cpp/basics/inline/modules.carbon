// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/inline/modules.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/inline/modules.carbon

// --- fail_export_module_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: This diagnostic isn't very good.
import Cpp inline '''
// CHECK:STDERR:        ╭─┤ fail_export_module_in_inline_cpp.carbon:[[@LINE+23]]:8
// CHECK:STDERR: ->     │ export module Foo;
// CHECK:STDERR:        ·        ━━┳━━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: module declaration must not come from an #include directive
// CHECK:STDERR:        ·             ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_export_module_in_inline_cpp.carbon:[[@LINE+16]]:8
// CHECK:STDERR: ->     │ export module Foo;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: module declaration must occur at the start of the translation
// CHECK:STDERR:        ·           ┃        unit
// CHECK:STDERR:        ·           ┃        [CppInteropParseError]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ <carbon Cpp imports>
// CHECK:STDERR:        │ ;
// CHECK:STDERR:        · ┬
// CHECK:STDERR:        · ╰──┤ add 'module;' to the start of the file to introduce a global module
// CHECK:STDERR:        · │  │ fragment
// CHECK:STDERR:        · │  │ [CppInteropParseNote]
// CHECK:STDERR:        · ╰────┤ insert `module; ` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
export module Foo;
''';

// --- fail_module_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: This diagnostic isn't very good.
import Cpp inline '''
// CHECK:STDERR:        ╭─┤ fail_module_in_inline_cpp.carbon:[[@LINE+30]]:1
// CHECK:STDERR: ->     │ module Bar;
// CHECK:STDERR:        · ━━━━━┳━━━━━
// CHECK:STDERR:        ·      ┗━━┫ error: module declaration must not come from an #include directive
// CHECK:STDERR:        ·         ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_module_in_inline_cpp.carbon:[[@LINE+23]]:1
// CHECK:STDERR: ->     │ module Bar;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: module declaration must occur at the start of the translation unit
// CHECK:STDERR:        ·    ┃        [CppInteropParseError]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ <carbon Cpp imports>
// CHECK:STDERR:        │ ;
// CHECK:STDERR:        · ┬
// CHECK:STDERR:        · ╰──┤ add 'module;' to the start of the file to introduce a global module
// CHECK:STDERR:        · │  │ fragment
// CHECK:STDERR:        · │  │ [CppInteropParseNote]
// CHECK:STDERR:        · ╰────┤ insert `module; ` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_module_in_inline_cpp.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ module Bar;
// CHECK:STDERR:        ·        ┳━━
// CHECK:STDERR:        ·        ┗━━┫ error: module name 'Foo' specified on command line does not match
// CHECK:STDERR:        ·           ┃        name of module
// CHECK:STDERR:        ·           ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
module Bar;
''';

// --- fail_global_module_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: This diagnostic isn't very good.
import Cpp inline '''
// CHECK:STDERR:        ╭─┤ fail_global_module_in_inline_cpp.carbon:[[@LINE+15]]:1
// CHECK:STDERR: ->     │ module;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: module declaration must not come from an #include directive
// CHECK:STDERR:        ·       ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_global_module_in_inline_cpp.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ module;
// CHECK:STDERR:        · ┳━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: 'module;' introducing a global module fragment can appear only at
// CHECK:STDERR:        ·    ┃        the start of the translation unit
// CHECK:STDERR:        ·    ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
module;

int n;

// CHECK:STDERR:        ╭─┤ fail_global_module_in_inline_cpp.carbon:[[@LINE+30]]:1
// CHECK:STDERR: ->     │ module Baz;
// CHECK:STDERR:        · ━━━━━┳━━━━━
// CHECK:STDERR:        ·      ┗━━┫ error: module declaration must not come from an #include directive
// CHECK:STDERR:        ·         ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_global_module_in_inline_cpp.carbon:[[@LINE+23]]:1
// CHECK:STDERR: ->     │ module Baz;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: module declaration must occur at the start of the translation unit
// CHECK:STDERR:        ·    ┃        [CppInteropParseError]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ <carbon Cpp imports>
// CHECK:STDERR:        │ ;
// CHECK:STDERR:        · ┬
// CHECK:STDERR:        · ╰──┤ add 'module;' to the start of the file to introduce a global module
// CHECK:STDERR:        · │  │ fragment
// CHECK:STDERR:        · │  │ [CppInteropParseNote]
// CHECK:STDERR:        · ╰────┤ insert `module; ` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_global_module_in_inline_cpp.carbon:[[@LINE+8]]:8
// CHECK:STDERR: ->     │ module Baz;
// CHECK:STDERR:        ·        ┳━━
// CHECK:STDERR:        ·        ┗━━┫ error: module name 'Foo' specified on command line does not match
// CHECK:STDERR:        ·           ┃        name of module
// CHECK:STDERR:        ·           ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
module Baz;
''';

// --- fail_import_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: Also test that a valid module import works.
import Cpp inline '''
// CHECK:STDERR:        ╭─┤ fail_import_in_inline_cpp.carbon:[[@LINE+7]]:8
// CHECK:STDERR: ->     │ import "file_that_does_not_exist.h";
// CHECK:STDERR:        ·        ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: 'file_that_does_not_exist.h' file not found
// CHECK:STDERR:        ·                         ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import "file_that_does_not_exist.h";
''';
