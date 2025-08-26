// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/snegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/snegate.carbon

// --- int_negate.carbon

library "[[@TEST_NAME]]";

fn Negate(a: i32) -> i32 = "int.snegate";

var arr: array(i32, Negate(Negate(123)));
let arr_p: array(i32, 123)* = &arr;

let n: i32 = Negate(1);

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Negate(a);
  //@dump-sem-ir-end
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.snegate";
fn Sub(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.ssub";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Negate(0)) as Expect(0);
  Test(Negate(1)) as Expect(Sub(0, 1));
  Test(Negate(Sub(0, 0x8000_0000_0000_0000))) as Expect(0x8000_0000_0000_0000);
}

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.snegate";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(i32, TooFew());
// CHECK:STDERR:                         ^~~~~~~~
// CHECK:STDERR:
var too_few: array(i32, TooFew());

fn RuntimeCallIsValidTooFew() -> i32 {
  return TooFew();
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(i32, TooMany(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(i32, TooMany(1, 2));

fn RuntimeCallIsValidTooMany(a: i32, b: i32) -> i32 {
  return TooMany(a, b);
}

// --- fail_bad_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.snegate";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(i32, BadReturnType(1));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(i32, BadReturnType(1));

fn RuntimeCallIsValidBadReturnType(a: i32) -> bool {
  return BadReturnType(a);
}

// --- fail_bad_call.carbon

library "[[@TEST_NAME]]";

fn Negate(a: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bad_call.carbon:[[@LINE+7]]:26: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(i32, Negate(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~
// CHECK:STDERR: fail_bad_call.carbon:[[@LINE-5]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn Negate(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(i32, Negate(1, 2));

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn Negate(a: i32) -> i32 = "int.snegate";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFF_FFFF));

// -INT_MIN is too large for i32.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in negation of -2147483648 [CompileTimeIntegerNegateOverflow]
// CHECK:STDERR: let b: i32 = Negate(-0x8000_0000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Negate(-0x8000_0000);

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Negate.type.15b: type = fn_type @Negate.loc4 [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type.15b = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type.15b = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %Negate.call: init %i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc13_19.1: %i32 = value_of_initializer %Negate.call
// CHECK:STDOUT:   %.loc13_19.2: %i32 = converted %Negate.call, %.loc13_19.1
// CHECK:STDOUT:   return %.loc13_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
