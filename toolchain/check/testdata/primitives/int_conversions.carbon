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

// --- fail_narrow.carbon

library "[[@TEST_NAME]]";

fn TestFailures() {
  var a: i64 = 1;
  // CHECK:STDERR: fail_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `i64` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused b: i32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow.carbon:[[@LINE+4]]:3: note: type `i64` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused b: i32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: i32 = a;

  var w: u64 = 1;
  // CHECK:STDERR: fail_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `u64` to `u32` [ConversionFailure]
  // CHECK:STDERR:   var unused x: u32 = w;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow.carbon:[[@LINE+4]]:3: note: type `u64` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused x: u32 = w;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused x: u32 = w;
}

// --- cpp_compat.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var c: i32 = 1;
  var unused d: Core.CppCompat.Long32 = c;
}

// --- fail_todo_cpp_compat_widen.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  var a: Core.CppCompat.Long32 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+11]]:17: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(33)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:   var unused b: i33 = a;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.Long32` to `i33` [ConversionFailure]
  // CHECK:STDERR:   var unused b: i33 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.Long32` does not implement interface `Core.ImplicitAs(i33)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused b: i33 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: i33 = a;

  var c: Core.CppCompat.LongLong64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+11]]:17: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:   var unused d: i65 = c;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.LongLong64` to `i65` [ConversionFailure]
  // CHECK:STDERR:   var unused d: i65 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.LongLong64` does not implement interface `Core.ImplicitAs(i65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused d: i65 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused d: i65 = c;

  var e: Core.CppCompat.ULong32 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+11]]:17: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(33)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:   var unused f: u33 = e;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.ULong32` to `u33` [ConversionFailure]
  // CHECK:STDERR:   var unused f: u33 = e;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.ULong32` does not implement interface `Core.ImplicitAs(u33)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused f: u33 = e;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused f: u33 = e;

  var g: Core.CppCompat.ULongLong64 = 1;
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+11]]:17: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(65)` instead [IntWidthNotMultipleOf8]
  // CHECK:STDERR:   var unused h: u65 = g;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.ULongLong64` to `u65` [ConversionFailure]
  // CHECK:STDERR:   var unused h: u65 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_compat_widen.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.ULongLong64` does not implement interface `Core.ImplicitAs(u65)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused h: u65 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused h: u65 = g;
}

// --- fail_cpp_compat_narrow.carbon

library "[[@TEST_NAME]]";

fn TestCppCompat() {
  // Long32 is slightly wider than i32.
  var a: Core.CppCompat.Long32 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.Long32` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused b: i32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.Long32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused b: i32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: i32 = a;

  // LongLong64 is slightly wider than i64.
  var c: Core.CppCompat.LongLong64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.LongLong64` to `i64` [ConversionFailure]
  // CHECK:STDERR:   var unused d: i64 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.LongLong64` does not implement interface `Core.ImplicitAs(i64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused d: i64 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused d: i64 = c;

  // ULong32 is slightly wider than u32.
  var e: Core.CppCompat.ULong32 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.ULong32` to `u32` [ConversionFailure]
  // CHECK:STDERR:   var unused f: u32 = e;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.ULong32` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused f: u32 = e;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused f: u32 = e;

  // ULongLong64 is slightly wider than u64.
  var g: Core.CppCompat.ULongLong64 = 1;
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.CppCompat.ULongLong64` to `u64` [ConversionFailure]
  // CHECK:STDERR:   var unused h: u64 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_compat_narrow.carbon:[[@LINE+4]]:3: note: type `Core.CppCompat.ULongLong64` does not implement interface `Core.ImplicitAs(u64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused h: u64 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused h: u64 = g;
}

// --- widen_sign_change.carbon

library "[[@TEST_NAME]]";

fn TestWidenSign() {
  var e: u32 = 1;
  var unused f: i64 = e;
}

// --- fail_narrow_sign_change.carbon

library "[[@TEST_NAME]]";

fn TestMixedSign() {
  var a: i32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `i32` to `u32` [ConversionFailure]
  // CHECK:STDERR:   var unused b: u32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+4]]:3: note: type `i32` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused b: u32 = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: u32 = a;

  var c: u32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `u32` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused d: i32 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+4]]:3: note: type `u32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused d: i32 = c;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused d: i32 = c;

  var g: i32 = 1;
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `i32` to `u64` [ConversionFailure]
  // CHECK:STDERR:   var unused h: u64 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_sign_change.carbon:[[@LINE+4]]:3: note: type `i32` does not implement interface `Core.ImplicitAs(u64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused h: u64 = g;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
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

// --- fail_narrow_arbitrary_width.carbon

library "[[@TEST_NAME]]";

fn TestArbitraryWidth() {
  var a: Core.Int(33) = 1;
  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `i33` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused b: Core.Int(32) = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+4]]:3: note: type `i33` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused b: Core.Int(32) = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: Core.Int(32) = a;

  var d: Core.UInt(33) = 1;
  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `u33` to `u32` [ConversionFailure]
  // CHECK:STDERR:   var unused e: Core.UInt(32) = d;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+4]]:3: note: type `u33` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused e: Core.UInt(32) = d;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused e: Core.UInt(32) = d;

  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `i33` to `u34` [ConversionFailure]
  // CHECK:STDERR:   var unused g: Core.UInt(34) = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_narrow_arbitrary_width.carbon:[[@LINE+4]]:3: note: type `i33` does not implement interface `Core.ImplicitAs(u34)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused g: Core.UInt(34) = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused g: Core.UInt(34) = a;
}
