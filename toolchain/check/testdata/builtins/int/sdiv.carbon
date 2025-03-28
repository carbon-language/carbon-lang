// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon

// --- int_div.carbon

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";

var arr: array(i32, Div(3, 2));
let arr_p: array(i32, 1)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Div(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

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

package FailDivByZero;

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
