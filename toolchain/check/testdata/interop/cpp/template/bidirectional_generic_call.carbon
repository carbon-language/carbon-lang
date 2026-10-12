// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/bidirectional_generic_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/bidirectional_generic_call.carbon

// --- member_function_calls_carbon.carbon

library "[[@TEST_NAME]]";

// Resolving `Make(C)` at the end of the file uses the constructor of
// `Wrapper(C)`, whose definition is only instantiated after that. The
// constructor calls back into Carbon, so `C` must still be usable from C++.
import Cpp inline '''
template<typename T>
struct Wrapper {
  Wrapper() { T::Callback(); }
};
''';

class C {
  fn Callback() {}
}

fn Make[T: type](unused x: T) {
  var unused w: Cpp.Wrapper(T);
}

fn Run() {
  Make({} as C);
}

// --- recursive_int.h

template<int I>
struct CarbonManualMultiply;

template<int I>
struct ManualMultiply {
  static auto Op(int x) -> int;
};

template<int I>
struct CarbonManualMultiply {
  static auto Op(int x) -> int;
};

template<int I>
auto ManualMultiply<I>::Op([[maybe_unused]] int x) -> int {
  static_assert(I != 0, "reached zero in C++");
  if constexpr (I == 0) {
    return 0;
  } else {
    return x + CarbonManualMultiply<I - 1>::Op(x);
  }
}

// --- recursive_int.carbon

library "[[@TEST_NAME]]";

import Cpp library "recursive_int.h";

class CarbonManualMultiply(template I: i32) {
  fn Op(x: i32) -> i32 {
    if (I == 0) {
      return 0;
    } else {
      return x + Cpp.ManualMultiply(I - 1).Op(x - 1);
    }
  }
}

fn Test() {
  let unused error: i32 = CarbonManualMultiply(9).Op(5);

  let unused okay: i32 = CarbonManualMultiply(10).Op(5);
}

