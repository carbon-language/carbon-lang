// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/neq.carbon

// --- float_neq.carbon

library "[[@TEST_NAME]]";

fn Neq(a: f64, b: f64) -> bool = "float.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Neq(1.0, 2.0) then True else False);
  false_ as (if Neq(1.0, 1.0) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return Neq(a, b);
  //@dump-sem-ir-end
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.neq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";

// CHECK:STDOUT: --- float_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [concrete]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %f64.d77, %b.param: %f64.d77) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: %Neq.type = name_ref Neq, file.%Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %f64.d77 = name_ref b, %b
// CHECK:STDOUT:   %Neq.call: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %Neq.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
