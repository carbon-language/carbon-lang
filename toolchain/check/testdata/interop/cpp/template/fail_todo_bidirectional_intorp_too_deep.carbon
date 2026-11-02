// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/fail_todo_bidirectional_intorp_too_deep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/fail_todo_bidirectional_intorp_too_deep.carbon

import Cpp;

class Wrap(T: Core.Copy) {
  var unwrap: T;
  impl as Core.Copy { fn Op(self) -> Self { return {.unwrap = self.unwrap}; } }
}
// TODO: C++ should be able to construct Wrap directly without this function.
fn MakeWrap[T: Core.Copy](t: T) -> Wrap(T) { return {.unwrap = t} as Wrap(T); }

fn CarbonSide[template T: type](p: T) -> T;

inline Cpp '''c++
template<typename T>
// CHECK:STDERR: fail_todo_bidirectional_intorp_too_deep.carbon:[[@LINE+2]]:6: error: semantics TODO: `maximum depth for bidirectional Carbon/C++ interop is 10` [SemanticsTodo]
// CHECK:STDERR:
auto AddTypeWrapper(T x) -> T {
    // Calls into to Carbon, adding a layer of Wrap(...) to the type T that
    // came from Carbon. This recursively generates an infinite number of
    // CarbonSide() specifics.
    return Carbon::CarbonSide(Carbon::MakeWrap(x)).unwrap;
  }
''';

fn CarbonSide[template T: type](p: T) -> T {
    return Cpp.AddTypeWrapper(p);
}

class C {
  adapt {};
  impl as Core.Copy { fn Op(unused self) -> Self { return {} as C; } }
}

fn Test() {
  var v: C = {} as C;
  CarbonSide(v);
}
