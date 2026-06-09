// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert_float_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert_float_checked.carbon

// --- int_to_float_checked.carbon
library "[[@TEST_NAME]]";

fn ConvertIntLiteralToFloatLiteral(a: Core.IntLiteral) -> Core.FloatLiteral = "int.convert_float_checked";
fn ConvertIntLiteralToF32(a: Core.IntLiteral) -> f32 = "int.convert_float_checked";

class Expect[T:! type](N:! T) {}
fn Test[T:! type](N:! T) -> Expect(N) { return {}; }

// --- identity.carbon
library "[[@TEST_NAME]]";
import library "int_to_float_checked";

fn F() {
  // Exact unsized float literal checked conversion.
  Test(ConvertIntLiteralToFloatLiteral(42) as f64) as Expect(42.0 as f64);

  // Direct exact sized float conversions.
  Test(ConvertIntLiteralToF32(12345)) as Expect(12345.0 as f32);
}

// --- fail_lossy.carbon
library "[[@TEST_NAME]]";
import library "int_to_float_checked";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_lossy.carbon:[[@LINE+4]]:18: error: integer value 123456789 cannot be represented exactly in floating-point type `f32` [IntLossyConversionToFloat]
// CHECK:STDERR: let lossy: f32 = ConvertIntLiteralToF32(123456789);
// CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let lossy: f32 = ConvertIntLiteralToF32(123456789);
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_lossy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.240: type = pattern_type %f32.c73 [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToF32.type: type = fn_type @ConvertIntLiteralToF32 [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToF32: %ConvertIntLiteralToF32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_123456789: Core.IntLiteral = int_value 123456789 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertIntLiteralToF32: %ConvertIntLiteralToF32.type = import_ref Main//int_to_float_checked, ConvertIntLiteralToF32, loaded [concrete = constants.%ConvertIntLiteralToF32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %lossy.patt: %pattern_type.240 = value_binding_pattern lossy [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:   %.loc9_50.1: %f32.c73 = value_of_initializer @__global_init.%ConvertIntLiteralToF32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc9_50.2: %f32.c73 = converted @__global_init.%ConvertIntLiteralToF32.call, %.loc9_50.1 [concrete = <error>]
// CHECK:STDOUT:   %lossy: %f32.c73 = value_binding lossy, %.loc9_50.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertIntLiteralToF32.ref: %ConvertIntLiteralToF32.type = name_ref ConvertIntLiteralToF32, imports.%Main.ConvertIntLiteralToF32 [concrete = constants.%ConvertIntLiteralToF32]
// CHECK:STDOUT:   %int_123456789: Core.IntLiteral = int_value 123456789 [concrete = constants.%int_123456789]
// CHECK:STDOUT:   %ConvertIntLiteralToF32.call: init %f32.c73 = call %ConvertIntLiteralToF32.ref(%int_123456789) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
