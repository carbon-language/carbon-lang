// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/unegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/unegate.carbon

// --- int_negate.carbon

library "[[@TEST_NAME]]";

fn Negate(a: u32) -> u32 = "int.unegate";

var arr: array(u32, Negate(Negate(123)));
let arr_p: array(u32, 123)* = &arr;

let n: u32 = Negate(1);

fn RuntimeCallIsValid(a: u32, b: u32) -> u32 {
  //@dump-sem-ir-begin
  return Negate(a);
  //@dump-sem-ir-end
}

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> u32 = "int.unegate";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(u32, TooFew());
// CHECK:STDERR:                         ^~~~~~~~
// CHECK:STDERR:
var too_few: array(u32, TooFew());

fn RuntimeCallIsValidTooFew() -> u32 {
  return TooFew();
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(u32, TooMany(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(u32, TooMany(1, 2));

fn RuntimeCallIsValidTooMany(a: u32, b: u32) -> u32 {
  return TooMany(a, b);
}

// --- fail_bad_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: u32) -> bool = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: u32) -> bool = "int.unegate";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(u32, BadReturnType(1));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(u32, BadReturnType(1));

fn RuntimeCallIsValidBadReturnType(a: u32) -> bool {
  return BadReturnType(a);
}

// --- fail_bad_call.carbon

library "[[@TEST_NAME]]";

fn Negate(a: u32) -> u32 = "int.unegate";

// CHECK:STDERR: fail_bad_call.carbon:[[@LINE+7]]:26: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(u32, Negate(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~
// CHECK:STDERR: fail_bad_call.carbon:[[@LINE-5]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn Negate(a: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(u32, Negate(1, 2));

// --- overflow.carbon

library "[[@TEST_NAME]]";

fn Negate(a: u32) -> u32 = "int.unegate";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn F() {
  // -(-INT_MAX) is INT_MAX.
  Test(Negate(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  Test(-(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  // -(-(INT_MAX + 1)) is `INT_MAX + 1`.
  Test(Negate(Negate(0x8000_0000))) as Expect(0x8000_0000);
  Test(-(Negate(0x8000_0000))) as Expect(0x8000_0000);
}

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %Negate.type.15b: type = fn_type @Negate.loc4 [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type.15b = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %u32, %b.param: %u32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type.15b = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %Negate.call: init %u32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc13_19.1: %u32 = value_of_initializer %Negate.call
// CHECK:STDOUT:   %.loc13_19.2: %u32 = converted %Negate.call, %.loc13_19.1
// CHECK:STDOUT:   return %.loc13_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
