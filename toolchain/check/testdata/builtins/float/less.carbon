// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/less.carbon

// --- float_less.carbon

fn Less(a: f64, b: f64) -> bool = "float.less";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1.0, 2.0) then True else False);
  false_ as (if Less(1.0, 1.0) then True else False);
  false_ as (if Less(1.0, 0.0) then True else False);
  true_ as (if Less(Negate(1.0), 0.0) then True else False);
  false_ as (if Less(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return Less(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [concrete = constants.%Less]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %Less.call: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_20.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc18_20.2: bool = converted %Less.call, %.loc18_20.1
// CHECK:STDOUT:   return %.loc18_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
