// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/add.carbon

// --- float_add.carbon

library "[[@TEST_NAME]]";

fn Add(a: f64, b: f64) -> f64 = "float.add";

fn RuntimeCallIsValid(a: f64, b: f64) -> f64 {
  //@dump-sem-ir-begin
  return Add(a, b);
  //@dump-sem-ir-end
}

var x: f64 = Add(2.2, 2.3);

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.add" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: f64) -> f64 = "float.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: f64) -> f64 = "float.add";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.add" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.add";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.add" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: f64, b: f64) -> bool = "float.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64, b: f64) -> bool = "float.add";
fn JustRight(a: f64, b: f64) -> f64 = "float.add";

fn RuntimeCallIsValidTooFew(a: f64) -> f64 {
  return TooFew(a);
}

fn RuntimeCallIsValidTooMany(a: f64, b: f64, c: f64) -> f64 {
  return TooMany(a, b, c);
}

fn RuntimeCallIsValidBadReturnType(a: f64, b: f64) -> bool {
  return BadReturnType(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Literal() -> type = "float_literal.make_type";
fn AddLiteral(a: Literal(), b: Literal()) -> Literal() = "float.add";

// --- literal_comptime.carbon

library "[[@TEST_NAME]]";
import library "literal";

let n: Literal() = AddLiteral(1.0, 2.0);

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";
import library "literal";

var m: Literal();
// CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+8]]:20: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let n: Literal() = AddLiteral(1.0, m);
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: literal.carbon:5:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn AddLiteral(a: Literal(), b: Literal()) -> Literal() = "float.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let n: Literal() = AddLiteral(1.0, m);

// CHECK:STDOUT: --- float_add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [concrete]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %f64.d77, %b.param: %f64.d77) -> %f64.d77 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %f64.d77 = name_ref b, %b
// CHECK:STDOUT:   %Add.call: init %f64.d77 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %f64.d77 = value_of_initializer %Add.call
// CHECK:STDOUT:   %.loc8_19.2: %f64.d77 = converted %Add.call, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
