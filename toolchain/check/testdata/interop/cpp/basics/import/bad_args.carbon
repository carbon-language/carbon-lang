// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --clang-arg=-fmodules --clang-arg=-fmodule-file=nonexistent.pcm
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/bad_args.carbon
// CHECK:STDERR: error: module file 'nonexistent.pcm' not found [CppInteropParseError]
// CHECK:STDERR:

// --- fail_use_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
int n;
''';
