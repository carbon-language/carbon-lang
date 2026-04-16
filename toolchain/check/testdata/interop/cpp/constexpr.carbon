// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
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

// --- pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
int i = 123;
constexpr int* _Nonnull ptr = &i;
''';

class C(V:! i32*) {}
fn F() -> C(Cpp.ptr);
let x: C(&Cpp.i) = F();

// --- function.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int f(int a, int b) { return a + b; }
''';

let a: array(i32, Cpp.f(1, 2)) = (1, 2, 3);

// --- function_bool_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int f(bool b) {
  return b ? 3 : 0;
}
''';

let a: array(i32, Cpp.f(true)) = (1, 2, 3);

// --- function_float_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int f(float b) {
  return static_cast<int>(b);
}
''';

let a: array(i32, Cpp.f(3.0)) = (1, 2, 3);

// --- function_return_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr bool f() {
  return true;
}
''';

musteval fn F(b: bool) -> i32 {
  return if b then 3 else 0;
}

let a: array(i32, F(Cpp.f())) = (1, 2, 3);

// --- fail_invalid_constant_eval.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int arr[1] = {0};
constexpr int f(int a) {
  return arr[a];
}
''';

// CHECK:STDERR: fail_invalid_constant_eval.carbon:[[@LINE+4]]:19: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: let a: array(i32, Cpp.f(5)) = (1, 2, 3);
// CHECK:STDERR:                   ^~~~~~~~
// CHECK:STDERR:
let a: array(i32, Cpp.f(5)) = (1, 2, 3);

// --- nonconstant_constexpr_call.carbon

library "[[@TEST_NAME]]";

import Cpp;

var a: i32 = 0;

inline Cpp '''
constexpr int f(const int &r) { return r; }
''';

// OK, runtime call.
var b: i32 = Cpp.f(a);

// --- fail_nonconstant_consteval_call.carbon

library "[[@TEST_NAME]]";

import Cpp;

var a: i32 = 0;

// TODO: This is failing when building the thunk, not when calling it.
inline Cpp '''
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+9]]:15: error: call to consteval function 'f' is not a constant expression [CppInteropParseError]
// CHECK:STDERR:    19 | consteval int f(const int &r) { return r; }
// CHECK:STDERR:       |               ^
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+6]]:15: note: function parameter 'r' with unknown value cannot be used in a constant expression [CppInteropParseNote]
// CHECK:STDERR:    19 | consteval int f(const int &r) { return r; }
// CHECK:STDERR:       |               ^
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+3]]:15: note: declared here [CppInteropParseNote]
// CHECK:STDERR:    19 | consteval int f(const int &r) { return r; }
// CHECK:STDERR:       |               ^
consteval int f(const int &r) { return r; }
''';

// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+11]]:14: note: in thunk for C++ function used here [InCppThunk]
// CHECK:STDERR: var b: i32 = Cpp.f(a);
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+7]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: var b: i32 = Cpp.f(a);
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE-10]]:15: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: consteval int f(const int &r) { return r; }
// CHECK:STDERR:               ^
// CHECK:STDERR:
var b: i32 = Cpp.f(a);
