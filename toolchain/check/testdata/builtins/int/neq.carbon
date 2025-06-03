// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/neq.carbon

// --- int_neq.carbon

fn Neq(a: i32, b: i32) -> bool = "int.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Neq(1, 1) then True else False);
  true_ as (if Neq(1, 2) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  //@dump-sem-ir-begin
  return Neq(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- int_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [concrete]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: %Neq.type = name_ref Neq, file.%Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.neq: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_19.1: bool = value_of_initializer %int.neq
// CHECK:STDOUT:   %.loc14_19.2: bool = converted %int.neq, %.loc14_19.1
// CHECK:STDOUT:   return %.loc14_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
