// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/constexpr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/constexpr.carbon

// --- bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr bool b = true;
''';

class C(B:! bool) {}
fn F() -> C(Cpp.b);
let x: C(true) = F();

// --- int.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int i = 123;
''';

class C(I:! i32) {}
fn F() -> C(Cpp.i);
let x: C(123) = F();

// --- float.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr float flt = 123.5;
''';

class C(V:! f32) {}
fn F() -> C(Cpp.flt);
let x: C(123.5) = F();

// --- function.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int f(int a, int b) { return a + b; }
''';

let a: array(i32, Cpp.f(1, 2)) = (1, 2, 3);

// --- fail_invalid_constant_eval.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int arr[1] = {0};
constexpr int f(int a) {
  return arr[a];
}
''';

// CHECK:STDERR: fail_invalid_constant_eval.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
// CHECK:STDERR: let a: array(i32, Cpp.f(5)) = (1, 2, 3);
// CHECK:STDERR:                   ^~~~~~~~
// CHECK:STDERR:
let a: array(i32, Cpp.f(5)) = (1, 2, 3);
