// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/convert_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/convert_checked.carbon

// --- float_ops.carbon

library "[[@TEST_NAME]]";

fn FloatLiteral() -> type = "float_literal.make_type";

// Size preserving
fn Float32ToFloat32(a: f32) -> f32 = "float.convert_checked";
fn Float64ToFloat64(a: f64) -> f64 = "float.convert_checked";
fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
    "float.convert_checked";

// Narrowing
fn Float64ToFloat32(a: f64) -> f32 = "float.convert_checked";
fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert_checked";
fn FloatLiteralToFloat64(a: FloatLiteral()) -> f64 = "float.convert_checked";

// Widening
fn Float32ToFloat64(a: f32) -> f64 = "float.convert_checked";
fn Float32ToFloatLiteral(a: f32) -> FloatLiteral() = "float.convert_checked";
fn Float64ToFloatLiteral(a: f64) -> FloatLiteral() = "float.convert_checked";

// --- runtime_call.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

//@dump-sem-ir-begin
let SizePreserving: f64 = Float64ToFloat64(1.0);
let Narrowing: f32 = Float64ToFloat32(1.0);
let Widening: f64 = Float32ToFloat64(1.0);
//@dump-sem-ir-end

// --- identity.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

let a: f32 = Float32ToFloat32(0.0);
let b: f32 = Float32ToFloat32(1.0);
let c: f32 = Float32ToFloat32(-1.0);
let d: FloatLiteral() = FloatLiteralToFloatLiteral(Float64ToFloatLiteral(-1.0));

// --- same_size.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

let a: f32 = Float32ToFloat32(1.0e10);
let b: f64 = Float64ToFloat64(1.0e10);

// --- truncate.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

let a: f32 = Float64ToFloat32(1.0);
let b: f32 = Float64ToFloat32(1.0e100);
let c: f32 = Float64ToFloat32(-1.0e100);

let lit_f32_one: f32 = FloatLiteralToFloat32(1.0);
let lit_f64_one: f64 = FloatLiteralToFloat64(1.0);

// --- extend.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

let a: f64 = Float32ToFloat64(1.0);
let b: f64 = Float32ToFloat64(1.0e30);
let c: f64 = Float32ToFloat64(-1.0e30);

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "float_ops";

let not_constant: f32 = 0.0;

let convert_not_constant_narrow: f32 = Float64ToFloat32(not_constant);
let convert_not_constant_same: f32 = Float32ToFloat32(not_constant);
let convert_not_constant_widen: f64 = Float32ToFloat64(not_constant);
