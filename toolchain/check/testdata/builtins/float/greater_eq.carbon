// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/greater_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/greater_eq.carbon

// --- float_greater_eq.carbon

fn GreaterEq(a: f64, b: f64) -> bool = "float.greater_eq";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if GreaterEq(1.0, 2.0) then True else False);
  true_ as (if GreaterEq(1.0, 1.0) then True else False);
  true_ as (if GreaterEq(1.0, 0.0) then True else False);
  false_ as (if GreaterEq(Negate(1.0), 0.0) then True else False);
  true_ as (if GreaterEq(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return GreaterEq(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_greater_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %GreaterEq.type: type = fn_type @GreaterEq [concrete]
// CHECK:STDOUT:   %GreaterEq: %GreaterEq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GreaterEq.ref: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [concrete = constants.%GreaterEq]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.greater_eq: init bool = call %GreaterEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_25.1: bool = value_of_initializer %float.greater_eq
// CHECK:STDOUT:   %.loc18_25.2: bool = converted %float.greater_eq, %.loc18_25.1
// CHECK:STDOUT:   return %.loc18_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
