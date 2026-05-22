// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=--std=c++26
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/thunk_cpp26.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/thunk_cpp26.carbon

// --- thunk_required.h

auto foo(short a) -> short;

// --- import_thunk_required.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk_required.h";

fn F() {
  // Placement operator new is `constexpr` in C++26. But that shouldn't cause
  // problems for us because we declare our own operator new.
  Cpp.foo(1 as i16);
}
