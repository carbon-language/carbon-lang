// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_add.carbon

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

var arr: [i32; Add(1, 2)];
let arr_p: [i32; 3]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.uadd".
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.uadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.uadd".
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.uadd".
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";
fn JustRight(a: i32, b: i32) -> i32 = "int.uadd";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: ERROR: Array bound is not a constant.
// CHECK:STDERR: var too_few: [i32; TooFew(1)];
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew(1)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: ERROR: Array bound is not a constant.
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: ERROR: Array bound is not a constant.
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1, 2)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1, 2)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:21: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: Calling function declared here.
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var bad_call: [i32; JustRight(1, 2, 3)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- overflow.carbon

package Overflow api;

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

// Overflow is OK.
let a: i32 = Add(0x7FFFFFFF, 0);
let b: i32 = Add(0x7FFFFFFF, 1);

// CHECK:STDOUT: --- int_add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Add = %Add
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Add: <function> = fn_decl @Add [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.ref: <function> = name_ref Add, %Add [template = %Add]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_19: init i32 = call %Add.ref(%.loc4_20, %.loc4_23) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 3] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 3] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 3] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 3] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 3]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 3]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: <function> = name_ref Add, file.%Add [template = file.%Add]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_13: init i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: i32 = value_of_initializer %.loc8_13
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %.loc8_13, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .TooFew = %TooFew
// CHECK:STDOUT:     .TooMany = %TooMany
// CHECK:STDOUT:     .BadReturnType = %BadReturnType
// CHECK:STDOUT:     .JustRight = %JustRight
// CHECK:STDOUT:     .too_few = %too_few
// CHECK:STDOUT:     .too_many = %too_many
// CHECK:STDOUT:     .bad_return_type = %bad_return_type
// CHECK:STDOUT:     .bad_call = %bad_call
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %TooFew: <function> = fn_decl @TooFew [template] {
// CHECK:STDOUT:     %a.loc8_11.1: i32 = param a
// CHECK:STDOUT:     @TooFew.%a: i32 = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     @TooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany: <function> = fn_decl @TooMany [template] {
// CHECK:STDOUT:     %a.loc13_12.1: i32 = param a
// CHECK:STDOUT:     @TooMany.%a: i32 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: i32 = param b
// CHECK:STDOUT:     @TooMany.%b: i32 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %c.loc13_28.1: i32 = param c
// CHECK:STDOUT:     @TooMany.%c: i32 = bind_name c, %c.loc13_28.1
// CHECK:STDOUT:     @TooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType: <function> = fn_decl @BadReturnType [template] {
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %b.loc18_26.1: i32 = param b
// CHECK:STDOUT:     @BadReturnType.%b: i32 = bind_name b, %b.loc18_26.1
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight: <function> = fn_decl @JustRight [template] {
// CHECK:STDOUT:     %a.loc19_14.1: i32 = param a
// CHECK:STDOUT:     @JustRight.%a: i32 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     %b.loc19_22.1: i32 = param b
// CHECK:STDOUT:     @JustRight.%b: i32 = bind_name b, %b.loc19_22.1
// CHECK:STDOUT:     @JustRight.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, %TooFew [template = %TooFew]
// CHECK:STDOUT:   %.loc25_27: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc25_26: init i32 = call %TooFew.ref(%.loc25_27)
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, %TooMany [template = %TooMany]
// CHECK:STDOUT:   %.loc30_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_32: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_35: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc30_28: init i32 = call %TooMany.ref(%.loc30_29, %.loc30_32, %.loc30_35)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, %BadReturnType [template = %BadReturnType]
// CHECK:STDOUT:   %.loc35_42: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc35_45: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc35_41: init bool = call %BadReturnType.ref(%.loc35_42, %.loc35_45)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %JustRight.ref: <function> = name_ref JustRight, %JustRight [template = %JustRight]
// CHECK:STDOUT:   %.loc43_31: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc43_34: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc43_37: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc43_30: init i32 = call %JustRight.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc43_39: type = array_type %.loc43_30, i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew: <function> = fn_decl @RuntimeCallTooFew [template] {
// CHECK:STDOUT:     %a.loc45_22.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: i32 = bind_name a, %a.loc45_22.1
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany: <function> = fn_decl @RuntimeCallTooMany [template] {
// CHECK:STDOUT:     %a.loc49_23.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc49_23.1
// CHECK:STDOUT:     %b.loc49_31.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc49_31.1
// CHECK:STDOUT:     %c.loc49_39.1: i32 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc49_39.1
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType: <function> = fn_decl @RuntimeCallBadReturnType [template] {
// CHECK:STDOUT:     %a.loc53_29.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc53_29.1
// CHECK:STDOUT:     %b.loc53_37.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc53_37.1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32, %c: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32, %b: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32, %b: i32) -> i32 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, file.%TooFew [template = file.%TooFew]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc46_16: init i32 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc46_19.1: i32 = value_of_initializer %.loc46_16
// CHECK:STDOUT:   %.loc46_19.2: i32 = converted %.loc46_16, %.loc46_19.1
// CHECK:STDOUT:   return %.loc46_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, file.%TooMany [template = file.%TooMany]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   %.loc50_17: init i32 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc50_26.1: i32 = value_of_initializer %.loc50_17
// CHECK:STDOUT:   %.loc50_26.2: i32 = converted %.loc50_17, %.loc50_26.1
// CHECK:STDOUT:   return %.loc50_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, file.%BadReturnType [template = file.%BadReturnType]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc54_23: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc54_29.1: bool = value_of_initializer %.loc54_23
// CHECK:STDOUT:   %.loc54_29.2: bool = converted %.loc54_23, %.loc54_29.1
// CHECK:STDOUT:   return %.loc54_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Add = %Add
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Add: <function> = fn_decl @Add [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.ref.loc7: <function> = name_ref Add, %Add [template = %Add]
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_17: init i32 = call %Add.ref.loc7(%.loc7_18, %.loc7_30) [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_32.1: i32 = value_of_initializer %.loc7_17 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_32.2: i32 = converted %.loc7_17, %.loc7_32.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.loc7: i32 = bind_name a, %.loc7_32.2
// CHECK:STDOUT:   %Add.ref.loc8: <function> = name_ref Add, %Add [template = %Add]
// CHECK:STDOUT:   %.loc8_18: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_30: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_17: init i32 = call %Add.ref.loc8(%.loc8_18, %.loc8_30) [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_32.1: i32 = value_of_initializer %.loc8_17 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_32.2: i32 = converted %.loc8_17, %.loc8_32.1 [template = constants.%.4]
// CHECK:STDOUT:   %b.loc8: i32 = bind_name b, %.loc8_32.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.uadd";
// CHECK:STDOUT:
