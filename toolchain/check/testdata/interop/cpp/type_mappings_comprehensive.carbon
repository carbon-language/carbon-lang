// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/type_mappings_comprehensive.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/type_mappings_comprehensive.carbon

// ============================================================================
// Comprehensive Type Mappings (Issue #5263)
// Based on carbon-interop Example 11: Complete Carbon ↔ C++ type system
// ============================================================================

// --- type_mappings.h

#include <cstdint>
#include <iostream>

// Signed integer type mappings
auto process_signed_char(signed char val) -> signed char;
auto process_short(short val) -> short;
auto process_int(int val) -> int;
auto process_long(long val) -> long;
auto process_long_long(long long val) -> long long;
auto process_int8_t(int8_t val) -> int8_t;
auto process_int16_t(int16_t val) -> int16_t;
auto process_int32_t(int32_t val) -> int32_t;
auto process_int64_t(int64_t val) -> int64_t;

// Unsigned integer type mappings
auto process_unsigned_char(unsigned char val) -> unsigned char;
auto process_unsigned_short(unsigned short val) -> unsigned short;
auto process_unsigned_int(unsigned int val) -> unsigned int;
auto process_unsigned_long(unsigned long val) -> unsigned long;
auto process_unsigned_long_long(unsigned long long val) -> unsigned long long;
auto process_uint8_t(uint8_t val) -> uint8_t;
auto process_uint16_t(uint16_t val) -> uint16_t;
auto process_uint32_t(uint32_t val) -> uint32_t;
auto process_uint64_t(uint64_t val) -> uint64_t;

// Floating-point type mappings
auto process_float(float val) -> float;
auto process_double(double val) -> double;
auto process_long_double(long double val) -> long double;

// Boolean and character types
auto process_bool(bool val) -> bool;
auto process_char(char val) -> char;
auto process_wchar_t(wchar_t val) -> wchar_t;

// Pointer and reference types (Issue #5426 integration)
auto process_int_ptr(int* val) -> int*;
auto process_int_ref(int& val) -> int&;
auto process_const_int_ref(const int& val) -> int;
auto process_double_ptr_ptr(double** val) -> double**;

// Array and string types
auto process_int_array(int arr[], size_t size) -> int;
auto process_const_char_ptr(const char* str) -> size_t;

// --- import_type_mappings.carbon

library "[[@TEST_NAME]]";

import Cpp library "type_mappings.h";

fn TestSignedIntegerMappings() {
  // Test i8 ↔ signed char / int8_t
  var val_i8: i8 = 127;
  var result_schar: i8 = Cpp.process_signed_char(val_i8);
  var result_int8: i8 = Cpp.process_int8_t(val_i8);
  
  // Test i16 ↔ short / int16_t
  var val_i16: i16 = 32767;
  var result_short: i16 = Cpp.process_short(val_i16);
  var result_int16: i16 = Cpp.process_int16_t(val_i16);
  
  // Test i32 ↔ int / int32_t
  var val_i32: i32 = 2147483647;
  var result_int: i32 = Cpp.process_int(val_i32);
  var result_int32: i32 = Cpp.process_int32_t(val_i32);
  
  // Test i64 ↔ long long / int64_t
  var val_i64: i64 = 9223372036854775807;
  var result_longlong: i64 = Cpp.process_long_long(val_i64);
  var result_int64: i64 = Cpp.process_int64_t(val_i64);
}

fn TestUnsignedIntegerMappings() {
  // Test u8 ↔ unsigned char / uint8_t
  var val_u8: u8 = 255;
  var result_uchar: u8 = Cpp.process_unsigned_char(val_u8);
  var result_uint8: u8 = Cpp.process_uint8_t(val_u8);
  
  // Test u16 ↔ unsigned short / uint16_t
  var val_u16: u16 = 65535;
  var result_ushort: u16 = Cpp.process_unsigned_short(val_u16);
  var result_uint16: u16 = Cpp.process_uint16_t(val_u16);
  
  // Test u32 ↔ unsigned int / uint32_t
  var val_u32: u32 = 4294967295;
  var result_uint: u32 = Cpp.process_unsigned_int(val_u32);
  var result_uint32: u32 = Cpp.process_uint32_t(val_u32);
  
  // Test u64 ↔ unsigned long long / uint64_t
  var val_u64: u64 = 18446744073709551615;
  var result_ulonglong: u64 = Cpp.process_unsigned_long_long(val_u64);
  var result_uint64: u64 = Cpp.process_uint64_t(val_u64);
}

fn TestFloatingPointMappings() {
  // Test f32 ↔ float
  var val_f32: f32 = 3.14159;
  var result_float: f32 = Cpp.process_float(val_f32);
  
  // Test f64 ↔ double
  var val_f64: f64 = 2.718281828459045;
  var result_double: f64 = Cpp.process_double(val_f64);
  
  // Test extended precision long double
  var result_long_double: f64 = Cpp.process_long_double(val_f64);
}

fn TestBooleanAndCharacterMappings() {
  // Test Bool ↔ bool
  var val_bool: Bool = true;
  var result_bool: Bool = Cpp.process_bool(val_bool);
  
  // Test character types
  var val_char: i8 = 65; // 'A'
  var result_char: i8 = Cpp.process_char(val_char);
}

fn TestPointerAndReferenceMappings() {
  // Test pointer types (from carbon-interop Issue #5426 solution)
  var val_int: i32 = 42;
  var result_ptr: i32* = Cpp.process_int_ptr(&val_int);
  
  // Test reference types (const and non-const)
  var result_ref: i32& = Cpp.process_int_ref(val_int);
  var result_const_ref: i32 = Cpp.process_const_int_ref(val_int);
  
  // Test double pointer
  var ptr_to_val: i32* = &val_int;
  var result_ptr_ptr: f64** = Cpp.process_double_ptr_ptr(&ptr_to_val);
}

fn TestArrayAndStringMappings() {
  // Test array processing
  var int_array: Array(i32) = Array(i32)();
  int_array.Push(1);
  int_array.Push(2);
  int_array.Push(3);
  int_array.Push(4);
  int_array.Push(5);
  
  var result_array: i32 = Cpp.process_int_array(int_array.Begin(), int_array.Size());
  
  // Test string processing
  var test_string: String = "Carbon C++ Interop Test";
  var result_strlen: UInt = Cpp.process_const_char_ptr(test_string.c_str());
}

fn TestComprehensiveTypeMappings() {
  TestSignedIntegerMappings();
  TestUnsignedIntegerMappings();
  TestFloatingPointMappings();
  TestBooleanAndCharacterMappings();
  TestPointerAndReferenceMappings();
  TestArrayAndStringMappings();
}

// --- type_mappings.cpp

#include "type_mappings.h"
#include <cstring>

// Signed integer implementations
auto process_signed_char(signed char val) -> signed char {
    std::cout << "Processing signed char: " << static_cast<int>(val) << "\n";
    return val * 2;
}

auto process_short(short val) -> short {
    std::cout << "Processing short: " << val << "\n";
    return val + 1;
}

auto process_int(int val) -> int {
    std::cout << "Processing int: " << val << "\n";
    return val - 1;
}

auto process_long(long val) -> long {
    std::cout << "Processing long: " << val << "\n";
    return val / 2;
}

auto process_long_long(long long val) -> long long {
    std::cout << "Processing long long: " << val << "\n";
    return val % 1000;
}

auto process_int8_t(int8_t val) -> int8_t {
    std::cout << "Processing int8_t: " << static_cast<int>(val) << "\n";
    return val ^ 0x0F;
}

auto process_int16_t(int16_t val) -> int16_t {
    std::cout << "Processing int16_t: " << val << "\n";
    return ~val;
}

auto process_int32_t(int32_t val) -> int32_t {
    std::cout << "Processing int32_t: " << val << "\n";
    return val << 1;
}

auto process_int64_t(int64_t val) -> int64_t {
    std::cout << "Processing int64_t: " << val << "\n";
    return val >> 1;
}

// Unsigned integer implementations
auto process_unsigned_char(unsigned char val) -> unsigned char {
    std::cout << "Processing unsigned char: " << static_cast<unsigned>(val) << "\n";
    return val + 1;
}

auto process_unsigned_short(unsigned short val) -> unsigned short {
    std::cout << "Processing unsigned short: " << val << "\n";
    return val * 2;
}

auto process_unsigned_int(unsigned int val) -> unsigned int {
    std::cout << "Processing unsigned int: " << val << "\n";
    return val / 3;
}

auto process_unsigned_long(unsigned long val) -> unsigned long {
    std::cout << "Processing unsigned long: " << val << "\n";
    return val % 1000;
}

auto process_unsigned_long_long(unsigned long long val) -> unsigned long long {
    std::cout << "Processing unsigned long long: " << val << "\n";
    return val & 0xFFFF;
}

auto process_uint8_t(uint8_t val) -> uint8_t {
    std::cout << "Processing uint8_t: " << static_cast<unsigned>(val) << "\n";
    return val | 0x80;
}

auto process_uint16_t(uint16_t val) -> uint16_t {
    std::cout << "Processing uint16_t: " << val << "\n";
    return val ^ 0xAAAA;
}

auto process_uint32_t(uint32_t val) -> uint32_t {
    std::cout << "Processing uint32_t: " << val << "\n";
    return val + 42;
}

auto process_uint64_t(uint64_t val) -> uint64_t {
    std::cout << "Processing uint64_t: " << val << "\n";
    return val - 1;
}

// Floating-point implementations
auto process_float(float val) -> float {
    std::cout << "Processing float: " << val << "\n";
    return val * 2.0f;
}

auto process_double(double val) -> double {
    std::cout << "Processing double: " << val << "\n";
    return val / 2.0;
}

auto process_long_double(long double val) -> long double {
    std::cout << "Processing long double: " << static_cast<double>(val) << "\n";
    return val + 1.0L;
}

// Boolean and character implementations
auto process_bool(bool val) -> bool {
    std::cout << "Processing bool: " << (val ? "true" : "false") << "\n";
    return !val;
}

auto process_char(char val) -> char {
    std::cout << "Processing char: '" << val << "'\n";
    return val + 1;
}

auto process_wchar_t(wchar_t val) -> wchar_t {
    std::cout << "Processing wchar_t: " << static_cast<int>(val) << "\n";
    return val + 1;
}

// Pointer and reference implementations
auto process_int_ptr(int* val) -> int* {
    if (val) {
        std::cout << "Processing int*: " << *val << "\n";
        *val *= 2;
    }
    return val;
}

auto process_int_ref(int& val) -> int& {
    std::cout << "Processing int&: " << val << "\n";
    val += 10;
    return val;
}

auto process_const_int_ref(const int& val) -> int {
    std::cout << "Processing const int&: " << val << "\n";
    return val * 3;
}

auto process_double_ptr_ptr(double** val) -> double** {
    if (val && *val) {
        std::cout << "Processing double**: " << **val << "\n";
    }
    return val;
}

// Array and string implementations
auto process_int_array(int arr[], size_t size) -> int {
    std::cout << "Processing int array of size: " << size << "\n";
    int sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += arr[i];
    }
    std::cout << "Array sum: " << sum << "\n";
    return sum;
}

auto process_const_char_ptr(const char* str) -> size_t {
    if (!str) return 0;
    size_t len = strlen(str);
    std::cout << "Processing string: \"" << str << "\" (length: " << len << ")\n";
    return len;
}

// CHECK:STDOUT: Processing signed char: 127
// CHECK:STDOUT: Processing int8_t: 127
// CHECK:STDOUT: Processing short: 32767
// CHECK:STDOUT: Processing int16_t: 32767
// CHECK:STDOUT: Processing int: 2147483647
// CHECK:STDOUT: Processing int32_t: 2147483647
// CHECK:STDOUT: Processing long long: 9223372036854775807
// CHECK:STDOUT: Processing int64_t: 9223372036854775807
// CHECK:STDOUT: Processing unsigned char: 255
// CHECK:STDOUT: Processing uint8_t: 255
// CHECK:STDOUT: Processing unsigned short: 65535
// CHECK:STDOUT: Processing uint16_t: 65535
// CHECK:STDOUT: Processing unsigned int: 4294967295
// CHECK:STDOUT: Processing uint32_t: 4294967295
// CHECK:STDOUT: Processing unsigned long long: 18446744073709551615
// CHECK:STDOUT: Processing uint64_t: 18446744073709551615
// CHECK:STDOUT: Processing float: 3.14159
// CHECK:STDOUT: Processing double: 2.71828
// CHECK:STDOUT: Processing long double: 2.71828
// CHECK:STDOUT: Processing bool: true
// CHECK:STDOUT: Processing char: 'A'
// CHECK:STDOUT: Processing int*: 42
// CHECK:STDOUT: Processing int&: 42
// CHECK:STDOUT: Processing const int&: 52
// CHECK:STDOUT: Processing int array of size: 5
// CHECK:STDOUT: Array sum: 15
// CHECK:STDOUT: Processing string: "Carbon C++ Interop Test" (length: 23) 