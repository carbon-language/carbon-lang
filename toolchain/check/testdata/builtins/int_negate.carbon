// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.negate";

var arr: [i32; Negate(Negate(1))];
let arr_p: [i32; 1]* = &arr;

let n: i32 = Negate(1);

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.negate".
// CHECK:STDERR: fn TooFew() -> i32 = "int.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.negate".
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.negate".
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.negate";
fn JustRight(a: i32) -> i32 = "int.negate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: ERROR: Array bound is not a constant.
// CHECK:STDERR: var too_few: [i32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: ERROR: Array bound is not a constant.
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: ERROR: Array bound is not a constant.
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: Calling function declared here.
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooFew() -> i32 = "int.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: Calling function declared here.
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow api;

fn Negate(a: i32) -> i32 = "int.negate";
fn Sub(a: i32, b: i32) -> i32 = "int.sub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFFFFFF));

// -(-INT_MAX - 1) is too large for i32.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: ERROR: Integer overflow in negation of -2147483648.
// CHECK:STDERR: let b: i32 = Negate(Sub(Negate(0x7FFFFFFF), 1));
// CHECK:STDERR:              ^~~~~~~
let b: i32 = Negate(Sub(Negate(0x7FFFFFFF), 1));

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc2_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc2_11.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref.loc4_16: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %Negate.ref.loc4_23: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_29.1: init i32 = call %Negate.ref.loc4_23(%.loc4_30) [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_22.1: i32 = value_of_initializer %.loc4_29.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_29.2: i32 = converted %.loc4_29.1, %.loc4_22.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_22.2: init i32 = call %Negate.ref.loc4_16(%.loc4_29.2) [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_33: type = array_type %.loc4_22.2, i32 [template = constants.%.3]
// CHECK:STDOUT:   %arr.var: ref [i32; 1] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 1] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 1] [template = constants.%.4]
// CHECK:STDOUT:   %arr.ref: ref [i32; 1] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 1]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 1]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %Negate.ref.loc7: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc7_21: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_20.1: init i32 = call %Negate.ref.loc7(%.loc7_21) [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_23: i32 = value_of_initializer %.loc7_20.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_20.2: i32 = converted %.loc7_20.1, %.loc7_23 [template = constants.%.2]
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc7_20.2
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc9_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc9_16.1
// CHECK:STDOUT:     %b.loc9_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc9_24.1
// CHECK:STDOUT:     %return.var.loc9: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, file.%Negate [template = file.%Negate]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc10_16.1: init i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19: i32 = value_of_initializer %.loc10_16.1
// CHECK:STDOUT:   %.loc10_16.2: i32 = converted %.loc10_16.1, %.loc10_19
// CHECK:STDOUT:   return %.loc10_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
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
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany: <function> = fn_decl @TooMany [template] {
// CHECK:STDOUT:     %a.loc13_12.1: i32 = param a
// CHECK:STDOUT:     @TooMany.%a: i32 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: i32 = param b
// CHECK:STDOUT:     @TooMany.%b: i32 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType: <function> = fn_decl @BadReturnType [template] {
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %return.var.loc18: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight: <function> = fn_decl @JustRight [template] {
// CHECK:STDOUT:     %a.loc19_14.1: i32 = param a
// CHECK:STDOUT:     @JustRight.%a: i32 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     %return.var.loc19: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, %TooFew [template = %TooFew]
// CHECK:STDOUT:   %.loc25: init i32 = call %TooFew.ref()
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, %TooMany [template = %TooMany]
// CHECK:STDOUT:   %.loc30_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_32: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_28: init i32 = call %TooMany.ref(%.loc30_29, %.loc30_32)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, %BadReturnType [template = %BadReturnType]
// CHECK:STDOUT:   %.loc35_42: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc35_41: init bool = call %BadReturnType.ref(%.loc35_42)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %JustRight.ref: <function> = name_ref JustRight, %JustRight [template = %JustRight]
// CHECK:STDOUT:   %.loc44_31: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc44_34: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_30: init i32 = call %JustRight.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc44_36: type = array_type %.loc44_30, i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew: <function> = fn_decl @RuntimeCallTooFew [template] {
// CHECK:STDOUT:     %a.loc46_22.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: i32 = bind_name a, %a.loc46_22.1
// CHECK:STDOUT:     %return.var.loc46: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany: <function> = fn_decl @RuntimeCallTooMany [template] {
// CHECK:STDOUT:     %a.loc57_23.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc57_23.1
// CHECK:STDOUT:     %b.loc57_31.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc57_31.1
// CHECK:STDOUT:     %c.loc57_39.1: i32 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc57_39.1
// CHECK:STDOUT:     %return.var.loc57: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType: <function> = fn_decl @RuntimeCallBadReturnType [template] {
// CHECK:STDOUT:     %a.loc68_29.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc68_29.1
// CHECK:STDOUT:     %b.loc68_37.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc68_37.1
// CHECK:STDOUT:     %return.var.loc68: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, file.%TooFew [template = file.%TooFew]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc54_16.1: init i32 = call %TooFew.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc54_19: i32 = value_of_initializer %.loc54_16.1 [template = <error>]
// CHECK:STDOUT:   %.loc54_16.2: i32 = converted %.loc54_16.1, %.loc54_19 [template = <error>]
// CHECK:STDOUT:   return %.loc54_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, file.%TooMany [template = file.%TooMany]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   %.loc65_17.1: init i32 = call %TooMany.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc65_26: i32 = value_of_initializer %.loc65_17.1 [template = <error>]
// CHECK:STDOUT:   %.loc65_17.2: i32 = converted %.loc65_17.1, %.loc65_26 [template = <error>]
// CHECK:STDOUT:   return %.loc65_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, file.%BadReturnType [template = file.%BadReturnType]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc76_23.1: init bool = call %BadReturnType.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc76_29: bool = value_of_initializer %.loc76_23.1 [template = <error>]
// CHECK:STDOUT:   %.loc76_23.2: bool = converted %.loc76_23.1, %.loc76_29 [template = <error>]
// CHECK:STDOUT:   return %.loc76_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .Sub = %Sub
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc4_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc4_11.1
// CHECK:STDOUT:     %return.var.loc4: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub: <function> = fn_decl @Sub [template] {
// CHECK:STDOUT:     %a.loc5_8.1: i32 = param a
// CHECK:STDOUT:     @Sub.%a: i32 = bind_name a, %a.loc5_8.1
// CHECK:STDOUT:     %b.loc5_16.1: i32 = param b
// CHECK:STDOUT:     @Sub.%b: i32 = bind_name b, %b.loc5_16.1
// CHECK:STDOUT:     %return.var.loc5: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref.loc8_14: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %Negate.ref.loc8_21: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc8_28: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_27.1: init i32 = call %Negate.ref.loc8_21(%.loc8_28) [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_20.1: i32 = value_of_initializer %.loc8_27.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_27.2: i32 = converted %.loc8_27.1, %.loc8_20.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_20.2: init i32 = call %Negate.ref.loc8_14(%.loc8_27.2) [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_40: i32 = value_of_initializer %.loc8_20.2 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_20.3: i32 = converted %.loc8_20.2, %.loc8_40 [template = constants.%.1]
// CHECK:STDOUT:   %a.loc8: i32 = bind_name a, %.loc8_20.3
// CHECK:STDOUT:   %Negate.ref.loc14_14: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %Sub.ref: <function> = name_ref Sub, %Sub [template = %Sub]
// CHECK:STDOUT:   %Negate.ref.loc14_25: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc14_32: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_31.1: init i32 = call %Negate.ref.loc14_25(%.loc14_32) [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_45: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_24.1: i32 = value_of_initializer %.loc14_31.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_31.2: i32 = converted %.loc14_31.1, %.loc14_24.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_24.2: init i32 = call %Sub.ref(%.loc14_31.2, %.loc14_45) [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_20.1: i32 = value_of_initializer %.loc14_24.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_24.3: i32 = converted %.loc14_24.2, %.loc14_20.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_20.2: init i32 = call %Negate.ref.loc14_14(%.loc14_24.3) [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_48: i32 = value_of_initializer %.loc14_20.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_20.3: i32 = converted %.loc14_20.2, %.loc14_48 [template = constants.%.4]
// CHECK:STDOUT:   %b.loc14: i32 = bind_name b, %.loc14_20.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a: i32, %b: i32) -> i32 = "int.sub";
// CHECK:STDOUT:
