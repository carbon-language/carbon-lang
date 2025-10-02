// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/eq.carbon

// --- float_eq.carbon

fn Eq(a: f64, b: f64) -> bool = "float.eq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Eq(1.0, 1.0) then True else False);
  false_ as (if Eq(1.0, 2.0) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return Eq(a, b);
  //@dump-sem-ir-end
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: f64, b: f64) -> f64 = "float.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongResult(a: f64, b: f64) -> f64 = "float.eq";

// --- fail_literal.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_literal.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Eq(a: Core.FloatLiteral(), b: Core.FloatLiteral()) -> bool = "float.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Eq(a: Core.FloatLiteral(), b: Core.FloatLiteral()) -> bool = "float.eq";

// CHECK:STDOUT: --- float_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [concrete]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %f64.d77, %b.param: %f64.d77) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref: %Eq.type = name_ref Eq, file.%Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %f64.d77 = name_ref b, %b
// CHECK:STDOUT:   %Eq.call: init bool = call %Eq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %Eq.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
