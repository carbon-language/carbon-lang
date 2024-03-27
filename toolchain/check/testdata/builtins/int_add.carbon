// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_add.carbon

fn Add(a: i32, b: i32) -> i32 = "int.add";

var arr: [i32; Add(1, 2)];
let arr_p: [i32; 3]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.add".
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.add";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.add".
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.add";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.add".
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.add";
fn JustRight(a: i32, b: i32) -> i32 = "int.add";

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

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: Calling function declared here.
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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

// --- fail_overflow.carbon

package FailOverflow api;

fn Add(a: i32, b: i32) -> i32 = "int.add";

let a: i32 = Add(0x7FFFFFFF, 0);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: ERROR: Integer overflow in calculation 2147483647 + 1.
// CHECK:STDERR: let b: i32 = Add(0x7FFFFFFF, 1);
// CHECK:STDERR:              ^~~~
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
// CHECK:STDOUT:     .Add = %Add
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add: <function> = fn_decl @Add [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
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
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.add";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: <function> = name_ref Add, file.%Add [template = file.%Add]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_13.1: init i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19: i32 = value_of_initializer %.loc8_13.1
// CHECK:STDOUT:   %.loc8_13.2: i32 = converted %.loc8_13.1, %.loc8_19
// CHECK:STDOUT:   return %.loc8_13.2
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
// CHECK:STDOUT:   %TooFew: <function> = fn_decl @TooFew [template] {
// CHECK:STDOUT:     %a.loc8_11.1: i32 = param a
// CHECK:STDOUT:     @TooFew.%a: i32 = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany: <function> = fn_decl @TooMany [template] {
// CHECK:STDOUT:     %a.loc13_12.1: i32 = param a
// CHECK:STDOUT:     @TooMany.%a: i32 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: i32 = param b
// CHECK:STDOUT:     @TooMany.%b: i32 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %c.loc13_28.1: i32 = param c
// CHECK:STDOUT:     @TooMany.%c: i32 = bind_name c, %c.loc13_28.1
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType: <function> = fn_decl @BadReturnType [template] {
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %b.loc18_26.1: i32 = param b
// CHECK:STDOUT:     @BadReturnType.%b: i32 = bind_name b, %b.loc18_26.1
// CHECK:STDOUT:     %return.var.loc18: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight: <function> = fn_decl @JustRight [template] {
// CHECK:STDOUT:     %a.loc19_14.1: i32 = param a
// CHECK:STDOUT:     @JustRight.%a: i32 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     %b.loc19_22.1: i32 = param b
// CHECK:STDOUT:     @JustRight.%b: i32 = bind_name b, %b.loc19_22.1
// CHECK:STDOUT:     %return.var.loc19: ref i32 = var <return slot>
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
// CHECK:STDOUT:   %.loc44_31: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc44_34: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_37: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_30: init i32 = call %JustRight.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc44_39: type = array_type %.loc44_30, i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew: <function> = fn_decl @RuntimeCallTooFew [template] {
// CHECK:STDOUT:     %a.loc46_22.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: i32 = bind_name a, %a.loc46_22.1
// CHECK:STDOUT:     %return.var.loc46: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany: <function> = fn_decl @RuntimeCallTooMany [template] {
// CHECK:STDOUT:     %a.loc50_23.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc50_23.1
// CHECK:STDOUT:     %b.loc50_31.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc50_31.1
// CHECK:STDOUT:     %c.loc50_39.1: i32 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc50_39.1
// CHECK:STDOUT:     %return.var.loc50: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType: <function> = fn_decl @RuntimeCallBadReturnType [template] {
// CHECK:STDOUT:     %a.loc54_29.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc54_29.1
// CHECK:STDOUT:     %b.loc54_37.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc54_37.1
// CHECK:STDOUT:     %return.var.loc54: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32, %c: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32, %b: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32, %b: i32) -> i32 = "int.add";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, file.%TooFew [template = file.%TooFew]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc47_16.1: init i32 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc47_19: i32 = value_of_initializer %.loc47_16.1
// CHECK:STDOUT:   %.loc47_16.2: i32 = converted %.loc47_16.1, %.loc47_19
// CHECK:STDOUT:   return %.loc47_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, file.%TooMany [template = file.%TooMany]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   %.loc51_17.1: init i32 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc51_26: i32 = value_of_initializer %.loc51_17.1
// CHECK:STDOUT:   %.loc51_17.2: i32 = converted %.loc51_17.1, %.loc51_26
// CHECK:STDOUT:   return %.loc51_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, file.%BadReturnType [template = file.%BadReturnType]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc55_23.1: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc55_29: bool = value_of_initializer %.loc55_23.1
// CHECK:STDOUT:   %.loc55_23.2: bool = converted %.loc55_23.1, %.loc55_29
// CHECK:STDOUT:   return %.loc55_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
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
// CHECK:STDOUT:     .Add = %Add
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add: <function> = fn_decl @Add [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.ref.loc6: <function> = name_ref Add, %Add [template = %Add]
// CHECK:STDOUT:   %.loc6_18: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_30: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_17.1: init i32 = call %Add.ref.loc6(%.loc6_18, %.loc6_30) [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_32: i32 = value_of_initializer %.loc6_17.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_17.2: i32 = converted %.loc6_17.1, %.loc6_32 [template = constants.%.1]
// CHECK:STDOUT:   %a.loc6: i32 = bind_name a, %.loc6_17.2
// CHECK:STDOUT:   %Add.ref.loc10: <function> = name_ref Add, %Add [template = %Add]
// CHECK:STDOUT:   %.loc10_18: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_30: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_17.1: init i32 = call %Add.ref.loc10(%.loc10_18, %.loc10_30) [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_32: i32 = value_of_initializer %.loc10_17.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_17.2: i32 = converted %.loc10_17.1, %.loc10_32 [template = constants.%.4]
// CHECK:STDOUT:   %b.loc10: i32 = bind_name b, %.loc10_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.add";
// CHECK:STDOUT:
