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
// CHECK:STDERR: fail_export_module_in_inline_cpp.carbon:[[@LINE+7]]:8: error: module declaration must occur at the start of the translation unit [CppInteropParseError]
// CHECK:STDERR:    13 | export module Foo;
// CHECK:STDERR:       |        ^
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: add 'module;' to the start of the file to introduce a global module fragment [CppInteropParseNote]
// CHECK:STDERR:     1 | # 6 "fail_export_module_in_inline_cpp.carbon"
// CHECK:STDERR:       | ^
// CHECK:STDERR:
export module Foo;
''';

// --- fail_module_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: This diagnostic isn't very good.
import Cpp inline '''
// CHECK:STDERR: fail_module_in_inline_cpp.carbon:[[@LINE+11]]:1: error: module declaration must occur at the start of the translation unit [CppInteropParseError]
// CHECK:STDERR:    17 | module Foo;
// CHECK:STDERR:       | ^
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: add 'module;' to the start of the file to introduce a global module fragment [CppInteropParseNote]
// CHECK:STDERR:     1 | # 6 "fail_module_in_inline_cpp.carbon"
// CHECK:STDERR:       | ^
// CHECK:STDERR:
// CHECK:STDERR: fail_module_in_inline_cpp.carbon:[[@LINE+4]]:8: error: module 'Foo' not found [CppInteropParseError]
// CHECK:STDERR:    17 | module Foo;
// CHECK:STDERR:       |        ^~~~
// CHECK:STDERR:
module Foo;
''';

// --- fail_global_module_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: This diagnostic isn't very good.
import Cpp inline '''
// CHECK:STDERR: fail_global_module_in_inline_cpp.carbon:[[@LINE+4]]:1: error: 'module;' introducing a global module fragment can appear only at the start of the translation unit [CppInteropParseError]
// CHECK:STDERR:    10 | module;
// CHECK:STDERR:       | ^~~~~~~
// CHECK:STDERR:
module;

int n;

// CHECK:STDERR: fail_global_module_in_inline_cpp.carbon:[[@LINE+11]]:1: error: module declaration must occur at the start of the translation unit [CppInteropParseError]
// CHECK:STDERR:    25 | module Foo;
// CHECK:STDERR:       | ^
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: add 'module;' to the start of the file to introduce a global module fragment [CppInteropParseNote]
// CHECK:STDERR:     1 | # 6 "fail_global_module_in_inline_cpp.carbon"
// CHECK:STDERR:       | ^
// CHECK:STDERR:
// CHECK:STDERR: fail_global_module_in_inline_cpp.carbon:[[@LINE+4]]:8: error: module 'Foo' not found [CppInteropParseError]
// CHECK:STDERR:    25 | module Foo;
// CHECK:STDERR:       |        ^~~~
// CHECK:STDERR:
module Foo;
''';

// --- fail_import_in_inline_cpp.carbon

library "[[@TEST_NAME]]";

// TODO: Also test that a valid module import works.
import Cpp inline '''
// CHECK:STDERR: fail_import_in_inline_cpp.carbon:[[@LINE+4]]:8: error: 'file_that_does_not_exist.h' file not found [CppInteropParseError]
// CHECK:STDERR:    10 | import "file_that_does_not_exist.h";
// CHECK:STDERR:       |        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import "file_that_does_not_exist.h";
''';
