// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/float_conversions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/float_conversions.carbon

// --- test_helpers.carbon
library "[[@TEST_NAME]]";

class Expect[T:! type](N:! T) {}
fn Test[T:! type](N:! T) -> Expect(N) { return {}; }

// --- literal_conversions.carbon
library "[[@TEST_NAME]]";

let _: f32 = 1.25;

let _: Core.FloatLiteral() = 42;
let _: Core.IntLiteral() = 42.0 unsafe as Core.IntLiteral();

// --- int_to_float.carbon
library "[[@TEST_NAME]]";

import library "test_helpers";

fn F(x: i32, y: u32) {
  let _: f64 = x;
  let _: f64 = y;

  Test(12345 as f32) as Expect(12345.0 as f32);
  Test(-42 as f64) as Expect(-(42.0 as f64));
}

// --- int_literal_to_float.carbon
library "[[@TEST_NAME]]";

fn F() {
  let _: f32 = 42;
  let _: f64 = 1234567890123456;
}

// --- fail_int_literal_to_float_lossy.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_int_literal_to_float_lossy.carbon:[[@LINE+4]]:14: error: integer value 123456789 cannot be represented exactly in floating-point type `f32` [IntLossyConversionToFloat]
// CHECK:STDERR: let a: f32 = 123456789;
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: f32 = 123456789;

// CHECK:STDERR: fail_int_literal_to_float_lossy.carbon:[[@LINE+4]]:14: error: integer value 100000001 cannot be represented exactly in floating-point type `f32` [IntLossyConversionToFloat]
// CHECK:STDERR: let b: f32 = 100000001;
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let b: f32 = 100000001;

// CHECK:STDERR: fail_int_literal_to_float_lossy.carbon:[[@LINE+4]]:14: error: integer value 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 too large for floating-point type `f32` [IntTooLargeForFloatType]
// CHECK:STDERR: let c: f32 = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: f32 = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;

// --- float_unsafe_as_int.carbon
library "[[@TEST_NAME]]";

import library "test_helpers";

fn F() {
  // Float literal to Sized Integer explicit conversion via `unsafe as`.
  Test(12.34 unsafe as i32) as Expect(12 as i32);

  Test(42.0 unsafe as u32) as Expect(42 as u32);

  // Negative float to signed integer.
  // TODO: Also test converting a negative float literal once they are
  // supported.
  Test(-(1.99 as f64) unsafe as i8) as Expect(-1 as i8);

  // Float literal to int literal.
  Test(123.45 unsafe as Core.IntLiteral()) as Expect(123);
}

// --- fail_float_as_int.carbon
library "[[@TEST_NAME]]";

import library "test_helpers";

fn F(f: f32) {
  // Lossy explicit conversion using `as` not permitted.
  // CHECK:STDERR: fail_float_as_int.carbon:[[@LINE+7]]:16: error: cannot convert expression of type `Core.FloatLiteral` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   let _: i32 = 12.34 as i32;
  // CHECK:STDERR:                ^~~~~~~~~~~~
  // CHECK:STDERR: fail_float_as_int.carbon:[[@LINE+4]]:16: note: type `Core.FloatLiteral` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: i32 = 12.34 as i32;
  // CHECK:STDERR:                ^~~~~~~~~~~~
  // CHECK:STDERR:
  let _: i32 = 12.34 as i32;

  // CHECK:STDERR: fail_float_as_int.carbon:[[@LINE+7]]:16: error: cannot convert expression of type `f32` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   let _: i32 = f as i32;
  // CHECK:STDERR:                ^~~~~~~~
  // CHECK:STDERR: fail_float_as_int.carbon:[[@LINE+4]]:16: note: type `f32` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: i32 = f as i32;
  // CHECK:STDERR:                ^~~~~~~~
  // CHECK:STDERR:
  let _: i32 = f as i32;
}

// --- float_to_float.carbon
library "[[@TEST_NAME]]";

import library "test_helpers";

fn F(x: f32) {
  // Implicit conversion (widening f32 -> f64)
  let y: f64 = x;

  // Explicit conversion (widening f32 -> f64)
  let _: f64 = x as f64;

  // Explicit conversion (narrowing f64 -> f32)
  let _: f32 = y as f32;
}

// --- fail_float_to_float_narrowing.carbon
library "[[@TEST_NAME]]";

fn F(x: f64) {
  // Implicit narrowing conversion (f64 -> f32) is not permitted.
  // CHECK:STDERR: fail_float_to_float_narrowing.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `f64` to `f32` [ConversionFailure]
  // CHECK:STDERR:   let _: f32 = x;
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_float_to_float_narrowing.carbon:[[@LINE+4]]:16: note: type `f64` does not implement interface `Core.ImplicitAs(f32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: f32 = x;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  let _: f32 = x;
}
