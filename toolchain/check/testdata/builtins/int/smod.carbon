// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smod.carbon

// --- int_div.carbon

library "[[@TEST_NAME]]";

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

var arr: array(i32, Mod(5, 3));
let arr_p: array(i32, 2)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Mod(a, b);
  //@dump-sem-ir-end
}

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn Mod(a: i32, b: i32) -> i32 = "int.smod";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF % -1 is OK.
let a: i32 = Mod(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 % 1 is OK.
let b: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows, so -0x8000_0000 % -1 is disallowed, even though
// its result is representable.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `-2147483648 % -1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

library "[[@TEST_NAME]]";

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

// Remainder of division by zero is not defined.

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Mod(1, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: i32 = Mod(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Mod(0, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let b: i32 = Mod(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Mod.type.797: type = fn_type @Mod.1 [concrete]
// CHECK:STDOUT:   %Mod: %Mod.type.797 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref: %Mod.type.797 = name_ref Mod, file.%Mod.decl [concrete = constants.%Mod]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smod: init %i32 = call %Mod.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.smod
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.smod, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
