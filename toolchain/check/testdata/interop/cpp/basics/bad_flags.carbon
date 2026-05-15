// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --clang-arg=-include-pch --clang-arg=does_not_exist.pch
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/bad_flags.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/bad_flags.carbon
// CHECK:STDERR: error: unable to read PCH file does_not_exist.pch: 'No such file or directory' [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: error: PCH file 'does_not_exist.pch' not found [CppInteropParseError]
// CHECK:STDERR:

// --- fail_bad_args.carbon

import Cpp inline "";

// Nothing to see here. We're just checking that this does not crash.
