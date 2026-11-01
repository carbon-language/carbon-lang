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

// Resolving `Make({} as C)` at the end of the file uses the constructor of
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

// --- alternating_passes.carbon

library "[[@TEST_NAME]]";

// Each step needs the other language to instantiate or resolve something new:
//
// 1. `Run` needs the C++ function template `CallCarbonGeneric<Box>`.
// 2. `CallCarbonGeneric<Box>` needs the Carbon generic
//    `UseCppClassTemplate(Box)`.
// 3. `UseCppClassTemplate(Box)` needs the C++ class template
//    `UsesCarbonInterface<Box>`.
// 4. `UsesCarbonInterface<Box>::Get` needs `Box` to implement the Carbon
//    interface `HasValue`, by calling `GetValue(Box)`. This is the base case.
import Cpp inline '''
template<typename T>
struct UsesCarbonInterface {
  static auto Get(const T& t) -> int;
};
''';

interface HasValue {
  fn Value(self) -> i32;
}

class Box {
  var n: i32;

  impl as HasValue {
    fn Value(self) -> i32 { return self.n; }
  }
}

fn GetValue[T: HasValue](t: T) -> i32 {
  return t.Value();
}

fn UseCppClassTemplate[T: type](t: T) -> i32 {
  return Cpp.UsesCarbonInterface(T).Get(t);
}

inline Cpp '''
template<typename T>
auto UsesCarbonInterface<T>::Get(const T& t) -> int {
  return Carbon::GetValue(t);
}

template<typename T>
auto CallCarbonGeneric(const T& t) -> int {
  return Carbon::UseCppClassTemplate(t);
}
''';

fn Run() -> i32 {
  let b: Box = {.n = 42};
  return Cpp.CallCarbonGeneric(b);
}
