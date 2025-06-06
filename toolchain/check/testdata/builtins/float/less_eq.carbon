// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/less_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/less_eq.carbon

// --- float_less_eq.carbon

fn LessEq(a: f64, b: f64) -> bool = "float.less_eq";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if LessEq(1.0, 2.0) then True else False);
  true_ as (if LessEq(1.0, 1.0) then True else False);
  false_ as (if LessEq(1.0, 0.0) then True else False);
  true_ as (if LessEq(Negate(1.0), 0.0) then True else False);
  false_ as (if LessEq(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return LessEq(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_less_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [concrete]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LessEq.ref: %LessEq.type = name_ref LessEq, file.%LessEq.decl [concrete = constants.%LessEq]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.less_eq: init bool = call %LessEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_22.1: bool = value_of_initializer %float.less_eq
// CHECK:STDOUT:   %.loc18_22.2: bool = converted %float.less_eq, %.loc18_22.1
// CHECK:STDOUT:   return %.loc18_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
