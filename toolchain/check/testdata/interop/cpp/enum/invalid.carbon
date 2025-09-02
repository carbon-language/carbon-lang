// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/invalid.carbon

// --- fail_invalid_underlying_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
// CHECK:STDERR: fail_invalid_underlying_type.carbon:[[@LINE+4]]:20: error: unknown type name 'no_such_type' [CppInteropParseError]
// CHECK:STDERR:     9 | enum InvalidEnum : no_such_type {};
// CHECK:STDERR:       |                    ^
// CHECK:STDERR:
enum InvalidEnum : no_such_type {};
''';

// Use of an invalid enum should not cause the toolchain to crash or produce
// redundant diagnostics.
var x: Cpp.InvalidEnum;
