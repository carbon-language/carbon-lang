// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.snegate";

var arr: [i32; Negate(Negate(123))];
let arr_p: [i32; 123]* = &arr;

let n: i32 = Negate(1);

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.snegate".
// CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.snegate".
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "int.snegate".
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.snegate";
fn JustRight(a: i32) -> i32 = "int.snegate";

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
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: Calling function declared here.
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow api;

fn Negate(a: i32) -> i32 = "int.snegate";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";

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
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Negate = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 123 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -123 [template]
// CHECK:STDOUT:   %.4: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 123] [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.2: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc2_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc2_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref.loc4_16: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Negate.ref.loc4_23: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 123 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate.loc4_29: init i32 = call %Negate.ref.loc4_23(%.loc4_30) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_22.1: i32 = value_of_initializer %int.snegate.loc4_29 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_22.2: i32 = converted %int.snegate.loc4_29, %.loc4_22.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.snegate.loc4_22: init i32 = call %Negate.ref.loc4_16(%.loc4_22.2) [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_35: type = array_type %int.snegate.loc4_22, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 123] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 123] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 123 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_21: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_22: type = ptr_type [i32; 123] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 123] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_26: [i32; 123]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 123]* = bind_name arr_p, %.loc5_26
// CHECK:STDOUT:   %Negate.ref.loc7: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc7_21: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %int.snegate.loc7: init i32 = call %Negate.ref.loc7(%.loc7_21) [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_23.1: i32 = value_of_initializer %int.snegate.loc7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_23.2: i32 = converted %int.snegate.loc7, %.loc7_23.1 [template = constants.%.7]
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc9_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc9_16.1
// CHECK:STDOUT:     %b.loc9_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc9_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: Negate = name_ref Negate, file.%Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc10_19.2: i32 = converted %int.snegate, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TooFew: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: TooFew = struct_value () [template]
// CHECK:STDOUT:   %TooMany: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %struct.2: TooMany = struct_value () [template]
// CHECK:STDOUT:   %BadReturnType: type = fn_type @BadReturnType [template]
// CHECK:STDOUT:   %struct.3: BadReturnType = struct_value () [template]
// CHECK:STDOUT:   %JustRight: type = fn_type @JustRight [template]
// CHECK:STDOUT:   %struct.4: JustRight = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %RuntimeCallTooFew: type = fn_type @RuntimeCallTooFew [template]
// CHECK:STDOUT:   %struct.5: RuntimeCallTooFew = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallTooMany: type = fn_type @RuntimeCallTooMany [template]
// CHECK:STDOUT:   %struct.6: RuntimeCallTooMany = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType: type = fn_type @RuntimeCallBadReturnType [template]
// CHECK:STDOUT:   %struct.7: RuntimeCallBadReturnType = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:     .BadReturnType = %BadReturnType.decl
// CHECK:STDOUT:     .JustRight = %JustRight.decl
// CHECK:STDOUT:     .too_few = %too_few
// CHECK:STDOUT:     .too_many = %too_many
// CHECK:STDOUT:     .bad_return_type = %bad_return_type
// CHECK:STDOUT:     .bad_call = %bad_call
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew.decl
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany.decl
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %TooFew.decl: TooFew = fn_decl @TooFew [template = constants.%struct.1] {
// CHECK:STDOUT:     @TooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: TooMany = fn_decl @TooMany [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc13_12.1: i32 = param a
// CHECK:STDOUT:     @TooMany.%a: i32 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: i32 = param b
// CHECK:STDOUT:     @TooMany.%b: i32 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     @TooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: BadReturnType = fn_decl @BadReturnType [template = constants.%struct.3] {
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: JustRight = fn_decl @JustRight [template = constants.%struct.4] {
// CHECK:STDOUT:     %a.loc19_14.1: i32 = param a
// CHECK:STDOUT:     @JustRight.%a: i32 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     @JustRight.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooFew.ref: TooFew = name_ref TooFew, %TooFew.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %TooFew.call: init i32 = call %TooFew.ref()
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %TooMany.ref: TooMany = name_ref TooMany, %TooMany.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc30_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_32: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %TooMany.call: init i32 = call %TooMany.ref(%.loc30_29, %.loc30_32)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %BadReturnType.ref: BadReturnType = name_ref BadReturnType, %BadReturnType.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %JustRight.ref: JustRight = name_ref JustRight, %JustRight.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc44_31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_34: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.snegate: init i32 = call %JustRight.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc44_36: type = array_type %int.snegate, i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: RuntimeCallTooFew = fn_decl @RuntimeCallTooFew [template = constants.%struct.5] {
// CHECK:STDOUT:     %a.loc46_22.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: i32 = bind_name a, %a.loc46_22.1
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: RuntimeCallTooMany = fn_decl @RuntimeCallTooMany [template = constants.%struct.6] {
// CHECK:STDOUT:     %a.loc57_23.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc57_23.1
// CHECK:STDOUT:     %b.loc57_31.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc57_31.1
// CHECK:STDOUT:     %c.loc57_39.1: i32 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc57_39.1
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: RuntimeCallBadReturnType = fn_decl @RuntimeCallBadReturnType [template = constants.%struct.7] {
// CHECK:STDOUT:     %a.loc68_29.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc68_29.1
// CHECK:STDOUT:     %b.loc68_37.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc68_37.1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: TooFew = name_ref TooFew, file.%TooFew.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init i32 = call %TooFew.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc54_19.1: i32 = value_of_initializer %TooFew.call [template = <error>]
// CHECK:STDOUT:   %.loc54_19.2: i32 = converted %TooFew.call, %.loc54_19.1 [template = <error>]
// CHECK:STDOUT:   return %.loc54_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: TooMany = name_ref TooMany, file.%TooMany.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   %TooMany.call: init i32 = call %TooMany.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc65_26.1: i32 = value_of_initializer %TooMany.call [template = <error>]
// CHECK:STDOUT:   %.loc65_26.2: i32 = converted %TooMany.call, %.loc65_26.1 [template = <error>]
// CHECK:STDOUT:   return %.loc65_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: BadReturnType = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc76_29.1: bool = value_of_initializer %BadReturnType.call [template = <error>]
// CHECK:STDOUT:   %.loc76_29.2: bool = converted %BadReturnType.call, %.loc76_29.1 [template = <error>]
// CHECK:STDOUT:   return %.loc76_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Negate = struct_value () [template]
// CHECK:STDOUT:   %Sub: type = fn_type @Sub [template]
// CHECK:STDOUT:   %struct.2: Sub = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -2147483647 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc4_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc4_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: Sub = fn_decl @Sub [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc5_8.1: i32 = param a
// CHECK:STDOUT:     @Sub.%a: i32 = bind_name a, %a.loc5_8.1
// CHECK:STDOUT:     %b.loc5_16.1: i32 = param b
// CHECK:STDOUT:     @Sub.%b: i32 = bind_name b, %b.loc5_16.1
// CHECK:STDOUT:     @Sub.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref.loc8_14: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Negate.ref.loc8_21: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc8_28: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate.loc8_27: init i32 = call %Negate.ref.loc8_21(%.loc8_28) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_20.1: i32 = value_of_initializer %int.snegate.loc8_27 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_20.2: i32 = converted %int.snegate.loc8_27, %.loc8_20.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.snegate.loc8_20: init i32 = call %Negate.ref.loc8_14(%.loc8_20.2) [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.1: i32 = value_of_initializer %int.snegate.loc8_20 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.2: i32 = converted %int.snegate.loc8_20, %.loc8_40.1 [template = constants.%.2]
// CHECK:STDOUT:   %a.loc8: i32 = bind_name a, %.loc8_40.2
// CHECK:STDOUT:   %Negate.ref.loc14_14: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Sub.ref: Sub = name_ref Sub, %Sub.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Negate.ref.loc14_25: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc14_32: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate.loc14_31: init i32 = call %Negate.ref.loc14_25(%.loc14_32) [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_45: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_24.1: i32 = value_of_initializer %int.snegate.loc14_31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_24.2: i32 = converted %int.snegate.loc14_31, %.loc14_24.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.ssub: init i32 = call %Sub.ref(%.loc14_24.2, %.loc14_45) [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_20.1: i32 = value_of_initializer %int.ssub [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_20.2: i32 = converted %int.ssub, %.loc14_20.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.snegate.loc14_20: init i32 = call %Negate.ref.loc14_14(%.loc14_20.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_48.1: i32 = value_of_initializer %int.snegate.loc14_20 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_48.2: i32 = converted %int.snegate.loc14_20, %.loc14_48.1 [template = constants.%.5]
// CHECK:STDOUT:   %b.loc14: i32 = bind_name b, %.loc14_48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a: i32, %b: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
