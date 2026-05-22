// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/generic_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/generic_call.carbon

// --- header.h

struct X {};

template<typename T> struct S {};

// --- fail_todo_generic_call.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

fn F[T:! type](unused x: T) {
  // TODO: This should be treated as being template-dependent on T,
  // and we should perform the call during monomorphization.
  // CHECK:STDERR: fail_todo_generic_call.carbon:[[@LINE+4]]:23: error: semantics TODO: `unsupported type used as template argument` [SemanticsTodo]
  // CHECK:STDERR:   var unused v: Cpp.S(T);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  var unused v: Cpp.S(T);
}

fn G() {
  F({} as Cpp.X);
}
