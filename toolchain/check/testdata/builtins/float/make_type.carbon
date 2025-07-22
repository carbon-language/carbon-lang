// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn Float(size: i32) -> type = "float.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

var f: Float(64) = 0.0;

fn GetFloat(dyn_size: i32) -> type {
  //@dump-sem-ir-begin
  return Float(dyn_size);
  //@dump-sem-ir-end
}

// --- fail_invalid_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR: var invalid_float: Float(931);
// CHECK:STDERR:                    ^~~~~~~~~~
// CHECK:STDERR:
var invalid_float: Float(931);

// --- fail_dyn_size.carbon

library "[[@TEST_NAME]]";

import library "types";

var dyn_size: i32 = 64;
// CHECK:STDERR: fail_dyn_size.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var dyn: Float(dyn_size);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var dyn: Float(dyn_size);

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float: %Float.type = import_ref Main//types, Float, loaded [concrete = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size.param: %i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%Main.Float [concrete = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %Float.call: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc10_25.1: type = value_of_initializer %Float.call
// CHECK:STDOUT:   %.loc10_25.2: type = converted %Float.call, %.loc10_25.1
// CHECK:STDOUT:   return %.loc10_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
