// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/uadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/uadd.carbon

// --- int_add.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

var arr: array(i32, Add(1, 2));
let arr_p: array(i32, 3)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Add(a, b);
  //@dump-sem-ir-end
}

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.uadd";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(i32, TooFew(1));
// CHECK:STDERR:                         ^~~~~~~~~
// CHECK:STDERR:
var too_few: array(i32, TooFew(1));

fn RuntimeCallIsValidTooFew(a: i32) -> i32 {
  return TooFew(a);
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(i32, TooMany(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(i32, TooMany(1, 2, 3));

fn RuntimeCallIsValidTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

// --- fail_bad_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(i32, BadReturnType(1, 2));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(i32, BadReturnType(1, 2));

fn RuntimeCallIsValidBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- fail_bad_call.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

// CHECK:STDERR: fail_bad_call.carbon:[[@LINE+7]]:26: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(i32, Add(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~~~~~~
// CHECK:STDERR: fail_bad_call.carbon:[[@LINE-5]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn Add(a: i32, b: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(i32, Add(1, 2, 3));

// --- overflow.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

// Overflow is OK.
let a: i32 = Add(0x7FFFFFFF, 0);
let b: i32 = Add(0x7FFFFFFF, 1);

// CHECK:STDOUT: --- int_add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Add.type.b1f: type = fn_type @Add.1 [concrete]
// CHECK:STDOUT:   %Add: %Add.type.b1f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type.b1f = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.uadd: init %i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.uadd
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.uadd, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
