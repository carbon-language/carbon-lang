// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reference_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reference_params.carbon

// ============================================================================
// C++ Reference and Pointer Parameter Support (Issue #5426)
// ============================================================================

// --- reference_params.h

// C++ functions with reference parameters
auto test_int_ref(int& x) -> int;
auto test_int_const_ref(const int& x) -> int;
auto test_void_ref(int& x) -> void;

// C++ functions with pointer parameters  
auto test_int_ptr(int* x) -> int;
auto test_int_const_ptr(const int* x) -> int;
auto test_void_ptr(int* x) -> void;

// Mixed reference and pointer parameters
auto test_mixed(int& ref_param, int* ptr_param) -> int;

// --- import_reference_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "reference_params.h";

fn TestReferenceParams() {
  var x: i32 = 42;
  
  // Test C++ reference parameter support
  // References should map to Carbon addresses automatically
  var result1: i32 = Cpp.test_int_ref(x);
  var result2: i32 = Cpp.test_int_const_ref(x);
  Cpp.test_void_ref(x);
  
  // Test C++ pointer parameter support  
  // Pointers should accept Carbon address-of expressions
  var result3: i32 = Cpp.test_int_ptr(&x);
  var result4: i32 = Cpp.test_int_const_ptr(&x);
  Cpp.test_void_ptr(&x);
  
  // Test mixed reference and pointer parameters
  var result5: i32 = Cpp.test_mixed(x, &x);
}

// --- reference_params.cpp

#include "reference_params.h"

auto test_int_ref(int& x) -> int {
  return ++x;
}

auto test_int_const_ref(const int& x) -> int {
  return x * 2;
}

auto test_void_ref(int& x) -> void {
  x += 10;
}

auto test_int_ptr(int* x) -> int {
  return x ? (*x + 1) : 0;
}

auto test_int_const_ptr(const int* x) -> int {
  return x ? (*x * 3) : 0;
}

auto test_void_ptr(int* x) -> void {
  if (x) *x += 20;
}

auto test_mixed(int& ref_param, int* ptr_param) -> int {
  ref_param += 5;
  return ptr_param ? (ref_param + *ptr_param) : ref_param;
}

// CHECK:STDOUT: --- reference_params.h
// CHECK:STDOUT: 
// CHECK:STDOUT: // C++ functions with reference parameters
// CHECK:STDOUT: auto test_int_ref(int& x) -> int;
// CHECK:STDOUT: auto test_int_const_ref(const int& x) -> int;
// CHECK:STDOUT: auto test_void_ref(int& x) -> void;
// CHECK:STDOUT: 
// CHECK:STDOUT: // C++ functions with pointer parameters  
// CHECK:STDOUT: auto test_int_ptr(int* x) -> int;
// CHECK:STDOUT: auto test_int_const_ptr(const int* x) -> int;
// CHECK:STDOUT: auto test_void_ptr(int* x) -> void;
// CHECK:STDOUT: 
// CHECK:STDOUT: // Mixed reference and pointer parameters
// CHECK:STDOUT: auto test_mixed(int& ref_param, int* ptr_param) -> int;
// CHECK:STDOUT: 
// CHECK:STDOUT: --- import_reference_params.carbon
// CHECK:STDOUT: 
// CHECK:STDOUT: library "[[@TEST_NAME]]";
// CHECK:STDOUT: 
// CHECK:STDOUT: import Cpp library "reference_params.h";
// CHECK:STDOUT: 
// CHECK:STDOUT: fn TestReferenceParams() {
// CHECK:STDOUT:   var x: i32 = 42;
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test C++ reference parameter support
// CHECK:STDOUT:   // References should map to Carbon addresses automatically
// CHECK:STDOUT:   var result1: i32 = Cpp.test_int_ref(x);
// CHECK:STDOUT:   var result2: i32 = Cpp.test_int_const_ref(x);
// CHECK:STDOUT:   Cpp.test_void_ref(x);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test C++ pointer parameter support  
// CHECK:STDOUT:   // Pointers should accept Carbon address-of expressions
// CHECK:STDOUT:   var result3: i32 = Cpp.test_int_ptr(&x);
// CHECK:STDOUT:   var result4: i32 = Cpp.test_int_const_ptr(&x);
// CHECK:STDOUT:   Cpp.test_void_ptr(&x);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test mixed reference and pointer parameters
// CHECK:STDOUT:   var result5: i32 = Cpp.test_mixed(x, &x);
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: --- reference_params.cpp
// CHECK:STDOUT: 
// CHECK:STDOUT: #include "reference_params.h"
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_int_ref(int& x) -> int {
// CHECK:STDOUT:   return ++x;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_int_const_ref(const int& x) -> int {
// CHECK:STDOUT:   return x * 2;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_void_ref(int& x) -> void {
// CHECK:STDOUT:   x += 10;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_int_ptr(int* x) -> int {
// CHECK:STDOUT:   return x ? (*x + 1) : 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_int_const_ptr(const int* x) -> int {
// CHECK:STDOUT:   return x ? (*x * 3) : 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_void_ptr(int* x) -> void {
// CHECK:STDOUT:   if (x) *x += 20;
// CHECK:STDOUT: }
// CHECK:STDOUT: 
// CHECK:STDOUT: auto test_mixed(int& ref_param, int* ptr_param) -> int {
// CHECK:STDOUT:   ref_param += 5;
// CHECK:STDOUT:   return ptr_param ? (ref_param + *ptr_param) : ref_param;
// CHECK:STDOUT: } 