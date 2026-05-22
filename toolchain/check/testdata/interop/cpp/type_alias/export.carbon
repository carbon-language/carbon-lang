// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/type_alias/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/type_alias/export.carbon

// --- aliases.carbon
library "[[@TEST_NAME]]";

import Cpp;

alias A = i32;

class C {}
alias CA = C;

inline Cpp '''
// Valid redeclarations -- same type.
using T = Carbon::A;
using T = int;

// Valid redeclarations -- same type.
using U = Carbon::C;
using U = Carbon::CA;
''';

// Valid: Cpp.U is the same type as CA.
fn F(p: CA*) -> Cpp.U* {
  return p;
}
