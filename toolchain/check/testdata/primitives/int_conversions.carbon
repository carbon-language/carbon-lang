// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/int_conversions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/int_conversions.carbon

// --- valid.carbon

library "[[@TEST_NAME]]";

fn TestFits() {
  var a: i32 = 1;
  var unused b: i64 = a;

  var w: u32 = 1;
  var unused x: u64 = w;
}

// --- fail_narrow_i64_to_i32.carbon

library "[[@TEST_NAME]]";

fn TestFailures() {
  var a: i64 = 1;
  // CHECK:STDERR: fail_narrow_i64_to_i32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `i64` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_i64_to_i32.carbon:[[@LINE+2]]:3-19: note: type `i64` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused b: i32 = a;
}

// --- fail_narrow_u64_to_u32.carbon

library "[[@TEST_NAME]]";

fn TestFailures() {
  var w: u64 = 1;
  // CHECK:STDERR: fail_narrow_u64_to_u32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `u64` to `u32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_u64_to_u32.carbon:[[@LINE+2]]:3-19: note: type `u64` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused x: u32 = w;
}

// --- cpp_compat.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var a: i32 = 1;
  var unused b: Core.CppCompat.Long32 = a;

  var c: u32 = 1;
  var unused d: Core.CppCompat.ULong32 = c;

  var e: Core.CppCompat.Long64 = 1;
  var unused f: i64 = e;

  var g: Core.CppCompat.ULong64 = 1;
  var unused h: u64 = g;

  var i: i64 = 1;
  var unused j: Core.CppCompat.LongLong64 = i;

  var k: u64 = 1;
  var unused l: Core.CppCompat.ULongLong64 = k;
}

// --- fail_todo_cpp_compat_widen_long32_to_i33.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var a: Core.CppCompat.Long32 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long32_to_i33.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(33)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long32_to_i33.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.Long32` to `i33` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long32_to_i33.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.Long32` does not implement interface `Core.ImplicitAs(i33)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused b: i33 = a;
}

// --- fail_todo_cpp_compat_widen_long64_to_i65.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var c: Core.CppCompat.Long64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long64_to_i65.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long64_to_i65.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.Long64` to `i65` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_long64_to_i65.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.Long64` does not implement interface `Core.ImplicitAs(i65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused d: i65 = c;
}

// --- fail_todo_cpp_compat_widen_longlong64_to_i65.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var e: Core.CppCompat.LongLong64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_longlong64_to_i65.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_longlong64_to_i65.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.LongLong64` to `i65` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_longlong64_to_i65.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.LongLong64` does not implement interface `Core.ImplicitAs(i65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused f: i65 = e;
}

// --- fail_todo_cpp_compat_widen_ulong32_to_u33.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var g: Core.CppCompat.ULong32 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong32_to_u33.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(33)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong32_to_u33.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.ULong32` to `u33` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong32_to_u33.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.ULong32` does not implement interface `Core.ImplicitAs(u33)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused h: u33 = g;
}

// --- fail_todo_cpp_compat_widen_ulong64_to_u65.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var i: Core.CppCompat.ULong64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong64_to_u65.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong64_to_u65.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.ULong64` to `u65` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulong64_to_u65.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.ULong64` does not implement interface `Core.ImplicitAs(u65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused j: u65 = i;
}

// --- fail_todo_cpp_compat_widen_ulonglong64_to_u65.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var k: Core.CppCompat.ULongLong64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulonglong64_to_u65.carbon:[[@LINE+5]]:17-19: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulonglong64_to_u65.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.ULongLong64` to `u65` [ConversionFailure]
  // CHECK:STDERR: fail_todo_cpp_compat_widen_ulonglong64_to_u65.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.ULongLong64` does not implement interface `Core.ImplicitAs(u65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused l: u65 = k;
}

// --- fail_cpp_compat_narrow_long32_to_i32.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // Long32 is slightly wider than i32.
  var a: Core.CppCompat.Long32 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_long32_to_i32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.Long32` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_long32_to_i32.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.Long32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused b: i32 = a;
}

// --- fail_cpp_compat_narrow_i64_to_long64.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // i64 is slightly wider than Long64.
  var c: i64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_i64_to_long64.carbon:[[@LINE+3]]:3-37: error: cannot implicitly convert expression of type `i64` to `Core.CppCompat.Long64` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_i64_to_long64.carbon:[[@LINE+2]]:3-37: note: type `i64` does not implement interface `Core.ImplicitAs(Core.CppCompat.Long64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused d: Core.CppCompat.Long64 = c;
}

// --- fail_cpp_compat_narrow_longlong64_to_i64.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // LongLong64 is slightly wider than i64.
  var e: Core.CppCompat.LongLong64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_longlong64_to_i64.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.LongLong64` to `i64` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_longlong64_to_i64.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.LongLong64` does not implement interface `Core.ImplicitAs(i64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused f: i64 = e;
}

// --- fail_cpp_compat_narrow_ulong32_to_u32.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // ULong32 is slightly wider than u32.
  var g: Core.CppCompat.ULong32 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_ulong32_to_u32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.ULong32` to `u32` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_ulong32_to_u32.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.ULong32` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused h: u32 = g;
}

// --- fail_cpp_compat_narrow_u64_to_ulong64.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // u64 is slightly wider than ULong64.
  var i: u64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_u64_to_ulong64.carbon:[[@LINE+3]]:3-38: error: cannot implicitly convert expression of type `u64` to `Core.CppCompat.ULong64` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_u64_to_ulong64.carbon:[[@LINE+2]]:3-38: note: type `u64` does not implement interface `Core.ImplicitAs(Core.CppCompat.ULong64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused j: Core.CppCompat.ULong64 = i;
}

// --- fail_cpp_compat_narrow_ulonglong64_to_u64.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // ULongLong64 is slightly wider than u64.
  var k: Core.CppCompat.ULongLong64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow_ulonglong64_to_u64.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `Core.CppCompat.ULongLong64` to `u64` [ConversionFailure]
  // CHECK:STDERR: fail_cpp_compat_narrow_ulonglong64_to_u64.carbon:[[@LINE+2]]:3-19: note: type `Core.CppCompat.ULongLong64` does not implement interface `Core.ImplicitAs(u64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused l: u64 = k;
}

// --- widen_sign_change.carbon

library "[[@TEST_NAME]]";

fn TestWidenSign() {
  var e: u32 = 1;
  var unused f: i64 = e;
}

// --- fail_narrow_sign_change_i32_to_u32.carbon

library "[[@TEST_NAME]]";

fn TestMixedSign() {
  var a: i32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change_i32_to_u32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `i32` to `u32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_sign_change_i32_to_u32.carbon:[[@LINE+2]]:3-19: note: type `i32` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused b: u32 = a;
}

// --- fail_narrow_sign_change_u32_to_i32.carbon

library "[[@TEST_NAME]]";

fn TestMixedSign() {
  var c: u32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change_u32_to_i32.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `u32` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_sign_change_u32_to_i32.carbon:[[@LINE+2]]:3-19: note: type `u32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused d: i32 = c;
}

// --- fail_narrow_sign_change_i32_to_u64.carbon

library "[[@TEST_NAME]]";

fn TestMixedSign() {
  var g: i32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change_i32_to_u64.carbon:[[@LINE+3]]:3-19: error: cannot implicitly convert expression of type `i32` to `u64` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_sign_change_i32_to_u64.carbon:[[@LINE+2]]:3-19: note: type `i32` does not implement interface `Core.ImplicitAs(u64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused h: u64 = g;
}

// --- arbitrary_width.carbon

library "[[@TEST_NAME]]";

fn TestArbitraryWidth() {
  var a: Core.Int(33) = 1;
  var unused c: Core.Int(34) = a;

  var d: Core.UInt(33) = 1;
  var unused f: Core.UInt(34) = d;

  var unused h: Core.Int(34) = d;
}

// --- fail_narrow_arbitrary_width_i33_to_i32.carbon

library "[[@TEST_NAME]]";

fn TestArbitraryWidth() {
  var a: Core.Int(33) = 1;
  // CHECK:STDERR: fail_narrow_arbitrary_width_i33_to_i32.carbon:[[@LINE+3]]:3-28: error: cannot implicitly convert expression of type `i33` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_arbitrary_width_i33_to_i32.carbon:[[@LINE+2]]:3-28: note: type `i33` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused b: Core.Int(32) = a;
}

// --- fail_narrow_arbitrary_width_u33_to_u32.carbon

library "[[@TEST_NAME]]";

fn TestArbitraryWidth() {
  var d: Core.UInt(33) = 1;
  // CHECK:STDERR: fail_narrow_arbitrary_width_u33_to_u32.carbon:[[@LINE+3]]:3-29: error: cannot implicitly convert expression of type `u33` to `u32` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_arbitrary_width_u33_to_u32.carbon:[[@LINE+2]]:3-29: note: type `u33` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused e: Core.UInt(32) = d;
}

// --- fail_narrow_arbitrary_width_i33_to_u34.carbon

library "[[@TEST_NAME]]";

fn TestArbitraryWidth() {
  var a: Core.Int(33) = 1;
  // CHECK:STDERR: fail_narrow_arbitrary_width_i33_to_u34.carbon:[[@LINE+3]]:3-29: error: cannot implicitly convert expression of type `i33` to `u34` [ConversionFailure]
  // CHECK:STDERR: fail_narrow_arbitrary_width_i33_to_u34.carbon:[[@LINE+2]]:3-29: note: type `i33` does not implement interface `Core.ImplicitAs(u34)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  var unused g: Core.UInt(34) = a;
}
