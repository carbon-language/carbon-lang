// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/alias.carbon

// --- alias_identical_address.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn F();
alias G = F;

inline Cpp '''
// CHECK:STDERR: alias_identical_address.carbon:[[@LINE+4]]:15: error: static assertion failed due to requirement '&Carbon::F == &Carbon::F' [CppInteropParseError]
// CHECK:STDERR:     9 | static_assert(&Carbon::F == &Carbon::G);
// CHECK:STDERR:       |               ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
static_assert(&Carbon::F == &Carbon::G);
''';
