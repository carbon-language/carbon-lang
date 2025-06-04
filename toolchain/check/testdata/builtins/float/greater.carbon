// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/greater.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/greater.carbon

// --- float_greater.carbon

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Greater(1.0, 2.0) then True else False);
  false_ as (if Greater(1.0, 1.0) then True else False);
  true_ as (if Greater(1.0, 0.0) then True else False);
  false_ as (if Greater(Negate(1.0), 0.0) then True else False);
  true_ as (if Greater(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return Greater(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_greater.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Greater.ref: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.greater: init bool = call %Greater.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_23.1: bool = value_of_initializer %float.greater
// CHECK:STDOUT:   %.loc18_23.2: bool = converted %float.greater, %.loc18_23.1
// CHECK:STDOUT:   return %.loc18_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
