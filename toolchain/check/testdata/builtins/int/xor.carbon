// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/xor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/xor.carbon

// --- int_xor.carbon

fn Xor(a: i32, b: i32) -> i32 = "int.xor";

var arr: array(i32, Xor(12, 10));
let arr_p: array(i32, 6)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Xor(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- int_xor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Xor.type: type = fn_type @Xor [concrete]
// CHECK:STDOUT:   %Xor: %Xor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Xor.ref: %Xor.type = name_ref Xor, file.%Xor.decl [concrete = constants.%Xor]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.xor: init %i32 = call %Xor.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc9_19.1: %i32 = value_of_initializer %int.xor
// CHECK:STDOUT:   %.loc9_19.2: %i32 = converted %int.xor, %.loc9_19.1
// CHECK:STDOUT:   return %.loc9_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
