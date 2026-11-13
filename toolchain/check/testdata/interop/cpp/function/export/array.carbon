// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/array.carbon

// --- array_value_param.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn FArrayValueParam(a: array(i32, 5)) -> i32 { return a[2]; }

inline Cpp '''
int GArrayValueParam() {
  int arr[5];
  //@dump-sem-ir-begin
  return Carbon::FArrayValueParam(arr);
  //@dump-sem-ir-end
}

int HArrayValueParam() {
  //@dump-sem-ir-begin
  return Carbon::FArrayValueParam({1, 2, 3, 4, 5});
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
// CHECK:STDERR:        ╭─┤ fail_todo_array_var_param.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ fn FArrayVarParam(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                         ┗━━┫ error: semantics TODO: `by-var array parameter`
// CHECK:STDERR:        ·                            ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_array_var_param.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn FArrayVarParam(var a: array(i32, 5)) -> i32 { return a[2]; }
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: semantics TODO: `failed to map C++ type to Carbon`
// CHECK:STDERR:        ·                      ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn FArrayVarParam(var a: array(i32, 5)) -> i32 { return a[2]; }

inline Cpp '''
int GArrayVarParam() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_array_var_param.carbon:[[@LINE+25]]:10
  // CHECK:STDERR:        │ int GArrayVarParam() {
  // CHECK:STDERR:        ·     ┬
  // CHECK:STDERR:        ·     ╰──┤ 'GArrayVarParam' declared here [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return Carbon::FArrayVarParam({1, 2, 3, 4, 5});
  // CHECK:STDERR:        ·          ┳─────────────────────
  // CHECK:STDERR:        ·          ┗━━┫ error: no member named 'FArrayVarParam' in namespace 'Carbon'; did
  // CHECK:STDERR:        ·          │  ┃        you mean 'GArrayVarParam'?
  // CHECK:STDERR:        ·          │  ┃        [CppInteropParseError]
  // CHECK:STDERR:        ·          ╰────┤ replace with `GArrayVarParam` [CppInteropFixItReplacement]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_todo_array_var_param.carbon:[[@LINE+12]]:33
  // CHECK:STDERR:        │ int GArrayVarParam() {
  // CHECK:STDERR:        ·     ┬
  // CHECK:STDERR:        ·     ╰──┤ 'GArrayVarParam' declared here [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return Carbon::FArrayVarParam({1, 2, 3, 4, 5});
  // CHECK:STDERR:        ·                  ────────────── ┳━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                 ┗━━┫ error: too many arguments to function call,
  // CHECK:STDERR:        ·                                    ┃        expected 0, have 1
  // CHECK:STDERR:        ·                                    ┃        [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return Carbon::FArrayVarParam({1, 2, 3, 4, 5});
  //@dump-sem-ir-end
}
''';

// --- fail_todo_array_return.carbon
library "[[@TEST_NAME]]";

import Cpp;

// We do not currently support C++ calls to functions with array return
// types, as C++ does not have such return types.
// TODO: Support this, perhaps by returning a `std::array`.

// CHECK:STDERR:        ╭─┤ fail_todo_array_return.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn FArrayReturn() -> array(i32, 5) {
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: semantics TODO: `array return type`
// CHECK:STDERR:        ·                      ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn FArrayReturn() -> array(i32, 5) {
  return (1, 2, 3, 4, 5);
}

inline Cpp '''
int GArrayReturn() {
  // CHECK:STDERR:        ╭─┤ fail_todo_array_return.carbon:[[@LINE+13]]:16
  // CHECK:STDERR:        │ int GArrayReturn() {
  // CHECK:STDERR:        ·     ┬
  // CHECK:STDERR:        ·     ╰──┤ 'GArrayReturn' declared here [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   auto &&arr = Carbon::FArrayReturn();
  // CHECK:STDERR:        ·                ┳───────────────────
  // CHECK:STDERR:        ·                ┗━━┫ error: no member named 'FArrayReturn' in namespace 'Carbon';
  // CHECK:STDERR:        ·                │  ┃        did you mean 'GArrayReturn'?
  // CHECK:STDERR:        ·                │  ┃        [CppInteropParseError]
  // CHECK:STDERR:        ·                ╰────┤ replace with `GArrayReturn` [CppInteropFixItReplacement]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  auto &&arr = Carbon::FArrayReturn();
  // CHECK:STDERR:        ╭─┤ fail_todo_array_return.carbon:[[@LINE+7]]:13
  // CHECK:STDERR: ->     │   return arr[0];
  // CHECK:STDERR:        ·          ──╴╻╶
  // CHECK:STDERR:        ·             ┗━━┫ error: subscripted value is not an array, pointer, or vector
  // CHECK:STDERR:        ·                ┃        [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return arr[0];
}
''';
