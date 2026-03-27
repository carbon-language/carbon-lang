// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/cpp.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/cpp.carbon

// --- std_initializer_list_make.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T>
struct InitializerListPointerPointer {
  T* begin;
  T* end;
};

template<typename T>
struct InitializerListPointerSize {
  T* begin;
  decltype(sizeof(T)) size;
};
''';

fn MakePointerPointer(a: array(i32, 4)) -> Cpp.InitializerListPointerPointer(i32) = "cpp.std.initializer_list.make";
fn MakePointerSize(a: array(i32, 4)) -> Cpp.InitializerListPointerSize(i32) = "cpp.std.initializer_list.make";

fn TakePointerPointer(a: Cpp.InitializerListPointerPointer(i32));
fn TakePointerSize(a: Cpp.InitializerListPointerSize(i32));

fn Test() {
  // CHECK:STDERR: std_initializer_list_make.carbon:[[@LINE+7]]:41: error: semantics TODO: `BISH: init_inst: {kind: Converted, arg0: inst7E0003C4, arg1: inst7E0005EA, type: type(inst7E00002F)}` [SemanticsTodo]
  // CHECK:STDERR:   TakePointerPointer(MakePointerPointer((1, 2, 3, 4)));
  // CHECK:STDERR:                                         ^~~~~~~~~~~~
  // CHECK:STDERR: std_initializer_list_make.carbon:[[@LINE-10]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn MakePointerPointer(a: array(i32, 4)) -> Cpp.InitializerListPointerPointer(i32) = "cpp.std.initializer_list.make";
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakePointerPointer(MakePointerPointer((1, 2, 3, 4)));
  // CHECK:STDERR: std_initializer_list_make.carbon:[[@LINE+7]]:35: error: semantics TODO: `BISH: init_inst: {kind: Converted, arg0: inst7E0005FA, arg1: inst7E000625, type: type(inst7E00002F)}` [SemanticsTodo]
  // CHECK:STDERR:   TakePointerSize(MakePointerSize((1, 2, 3, 4)));
  // CHECK:STDERR:                                   ^~~~~~~~~~~~
  // CHECK:STDERR: std_initializer_list_make.carbon:[[@LINE-17]]:20: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn MakePointerSize(a: array(i32, 4)) -> Cpp.InitializerListPointerSize(i32) = "cpp.std.initializer_list.make";
  // CHECK:STDERR:                    ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakePointerSize(MakePointerSize((1, 2, 3, 4)));
}
