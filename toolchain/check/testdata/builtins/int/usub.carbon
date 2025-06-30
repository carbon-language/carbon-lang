// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/usub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/usub.carbon

// --- int_sub.carbon

library "[[@TEST_NAME]]";

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

var arr: array(i32, Sub(3, 2));
let arr_p: array(i32, 1)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Sub(a, b);
  //@dump-sem-ir-end
}

// --- overflow.carbon

library "[[@TEST_NAME]]";

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

let a: i32 = Sub(0, 0x7FFFFFFF);
let b: i32 = Sub(Sub(0, 0x7FFFFFFF), 1);
let c: i32 = Sub(Sub(0, 0x7FFFFFFF), 2);

// CHECK:STDOUT: --- int_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [concrete]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.usub
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.usub, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
