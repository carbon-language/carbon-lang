// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/fail_todo_bidirectional_intorp_too_deep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/fail_todo_bidirectional_intorp_too_deep.carbon

import Cpp;

fn ManualMultiply[template T: type](p: T*) -> T*;

inline Cpp '''c++
template<typename T>
auto ManualMultiply(T x) -> T {
    return *Carbon::ManualMultiply(&x);
  }
''';

fn ManualMultiply[template T: type](p: T*) -> T* {
    // CHECK:STDERR: fail_todo_bidirectional_intorp_too_deep.carbon:[[@LINE+2]]:32: error: semantics TODO: `maximum depth for bidirectional Carbon/C++ interop is 21` [SemanticsTodo]
    // CHECK:STDERR:
    return Cpp.ManualMultiply(p);
}

fn Test() {
  var v: i32 = 5;
  ManualMultiply(&v);
}
