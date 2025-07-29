// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/div.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/div.carbon

// --- float_div.carbon

library "[[@TEST_NAME]]";

fn Div(a: f64, b: f64) -> f64 = "float.div";

fn RuntimeCallIsValid(a: f64, b: f64) -> f64 {
  //@dump-sem-ir-begin
  return Div(a, b);
  //@dump-sem-ir-end
}

var a: f64 = Div(10.0, 2.5);
let b: f64 = Div(1.0, 0.0);
let c: f64 = Div(0.0, 0.0);

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.div" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: f64) -> f64 = "float.div";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: f64) -> f64 = "float.div";

fn RuntimeCallIsValidTooFew(a: f64) -> f64 {
  return TooFew(a);
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.div" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.div";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.div";

fn RuntimeCallIsValidTooMany(a: f64, b: f64, c: f64) -> f64 {
  return TooMany(a, b, c);
}

// --- fail_bad_return_type.carbon

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.div" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: f64, b: f64) -> bool = "float.div";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64, b: f64) -> bool = "float.div";

fn RuntimeCallIsValidBadReturnType(a: f64, b: f64) -> bool {
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- float_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [concrete]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: %Div.type = name_ref Div, file.%Div.decl [concrete = constants.%Div]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %Div.call: init f64 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: f64 = value_of_initializer %Div.call
// CHECK:STDOUT:   %.loc8_19.2: f64 = converted %Div.call, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
