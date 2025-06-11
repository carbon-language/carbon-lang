// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon

// --- int_div.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";

var arr: array(i32, Div(3, 2));
let arr_p: array(i32, 1)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Div(a, b);
  //@dump-sem-ir-end
}

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(-0x7FFF_FFFF, -1);

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(-0x8000_0000, 1);

// -0x8000_0000 / -1 overflows.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `-2147483648 / -1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let c: i32 = Div(-0x8000_0000, -1);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = Div(-0x8000_0000, -1);

// --- literal_no_overflow.carbon

library "[[@TEST_NAME]]";

fn Div(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sdiv";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Div(-0x8000_0000, -1)) as Expect(0x8000_0000);
  Test(Div(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_div_by_zero.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn DivLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sdiv";

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Div(1, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: i32 = Div(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Div(0, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let b: i32 = Div(0, 0);

// IntLiteral allows "overflow" by widening its representation, but not overflow to infinity.
// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:28: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let c: Core.IntLiteral() = DivLit(1, 0);
// CHECK:STDERR:                            ^~~~~~~~~~~~
// CHECK:STDERR:
let c: Core.IntLiteral() = DivLit(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:28: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let d: Core.IntLiteral() = DivLit(0, 0);
// CHECK:STDERR:                            ^~~~~~~~~~~~
// CHECK:STDERR:
let d: Core.IntLiteral() = DivLit(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Div.type.ab4: type = fn_type @Div.1 [concrete]
// CHECK:STDOUT:   %Div: %Div.type.ab4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: %Div.type.ab4 = name_ref Div, file.%Div.decl [concrete = constants.%Div]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sdiv: init %i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.sdiv
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.sdiv, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
