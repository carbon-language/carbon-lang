// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/array.carbon

// --- array_value_param.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn F(a: array(i32, 5)) -> i32 { return a[2]; }

inline Cpp '''
int G() {
  int arr[5];
  //@dump-sem-ir-begin
  return Carbon::F(arr);
  //@dump-sem-ir-end
}

int H() {
  //@dump-sem-ir-begin
  return Carbon::F({1, 2, 3, 4, 5});
  //@dump-sem-ir-end
}
''';

// --- fail_todo_array_var_param.carbon
library "[[@TEST_NAME]]";

import Cpp;

// We do not currently support C++ calls to functions with by-var array
// parameters, as C++ does not have such parameters.
// TODO: Support this, perhaps using Clang's `ArrayParameterType`, which is an
// HLSL extension, or by mapping to a `std::array` parameter.

// TODO: These diagnostics are very bad.
// CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+18]]:1: error: cannot copy value of type `array(i32, 5)` [CopyOfUncopyableType]
// CHECK:STDERR: fn F(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+15]]:1: note: type `array(i32, 5)` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+12]]:6: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn F(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+8]]:1: error: semantics TODO: `by-var array parameter` [SemanticsTodo]
// CHECK:STDERR: fn F(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+4]]:1: error: semantics TODO: `failed to map C++ type to Carbon` [SemanticsTodo]
// CHECK:STDERR: fn F(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(var a: array(i32, 5)) -> i32 { return a[2]; }

inline Cpp '''
int G() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_array_var_param.carbon:[[@LINE+4]]:18: error: no member named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    38 |   return Carbon::F({1, 2, 3, 4, 5});
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  return Carbon::F({1, 2, 3, 4, 5});
  //@dump-sem-ir-end
}
''';

// --- fail_todo_array_return.carbon
library "[[@TEST_NAME]]";

import Cpp;

// We do not currently support C++ calls to functions with array return
// types, as C++ does not have such return types.
// TODO: Support this, perhaps by returning a `std::array`.

// CHECK:STDERR: fail_todo_array_return.carbon:[[@LINE+4]]:1: error: semantics TODO: `array return type` [SemanticsTodo]
// CHECK:STDERR: fn F() -> array(i32, 5) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> array(i32, 5) {
  return (1, 2, 3, 4, 5);
}

inline Cpp '''
int G() {
  // CHECK:STDERR: fail_todo_array_return.carbon:[[@LINE+4]]:24: error: no member named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    23 |   auto &&arr = Carbon::F();
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR:
  auto &&arr = Carbon::F();
  return arr[0];
}
''';
