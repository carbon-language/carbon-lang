// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/pointer.carbon

// ============================================================================
// Basic pointer parameter
// ============================================================================

// --- int_ptr_param.h

auto take_int_ptr(int* p) -> void;

// --- import_int_ptr_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_ptr_param.h";

fn TestIntPtrParam() {
  //@dump-sem-ir-begin
  var x: i32 = 42;
  Cpp.take_int_ptr(&x);
  //@dump-sem-ir-end
}

// ============================================================================
// Basic pointer return type
// ============================================================================

// --- int_ptr_return.h

auto get_int_ptr() -> int*;

// --- import_int_ptr_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_ptr_return.h";

fn TestIntPtrReturn() {
  //@dump-sem-ir-begin
  var ptr: i32* = Cpp.get_int_ptr();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to pointer (nested)
// ============================================================================

// --- int_ptr_ptr.h

auto take_int_ptr_ptr(int** pp) -> void;

// --- import_int_ptr_ptr.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_ptr_ptr.h";

fn TestIntPtrPtr() {
  //@dump-sem-ir-begin
  var x: i32 = 42;
  var ptr: i32* = &x;
  Cpp.take_int_ptr_ptr(&ptr);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to struct
// ============================================================================

// --- struct_ptr.h

struct Point {
  int x;
  int y;
};

auto take_point_ptr(Point* p) -> void;

// --- import_struct_ptr.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_ptr.h";

fn TestStructPtr() {
  //@dump-sem-ir-begin
  var p: Cpp.Point = {.x = 10, .y = 20};
  Cpp.take_point_ptr(&p);
  //@dump-sem-ir-end
}

// ============================================================================
// Function with mixed pointer and non-pointer parameters
// ============================================================================

// --- mixed_params.h

auto mixed_function(int value, int* ptr, short another_value) -> void;

// --- import_mixed_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "mixed_params.h";

fn TestMixedParams() {
  //@dump-sem-ir-begin
  var x: i32 = 42;
  var y: i16 = 10;
  Cpp.mixed_function(5, &x, y);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to const (should work the same as regular pointer for now)
// ============================================================================

// --- const_ptr.h

auto take_const_int_ptr(const int* p) -> void;

// --- import_const_ptr.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_ptr.h";

fn TestConstPtr() {
  //@dump-sem-ir-begin
  var x: i32 = 42;
  Cpp.take_const_int_ptr(&x);
  //@dump-sem-ir-end
} 