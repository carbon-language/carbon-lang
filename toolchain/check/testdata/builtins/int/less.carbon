// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/less.carbon

// --- int_less.carbon

fn Less(a: i32, b: i32) -> bool = "int.less";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1, 2) then True else False);
  false_ as (if Less(1, 1) then True else False);
  false_ as (if Less(1, 0) then True else False);
  true_ as (if Less(Negate(1), 0) then True else False);
  false_ as (if Less(0, Negate(1)) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  //@dump-sem-ir-begin
  return Less(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- int_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [concrete = constants.%Less]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %Less.call: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_20.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc18_20.2: bool = converted %Less.call, %.loc18_20.1
// CHECK:STDOUT:   return %.loc18_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
